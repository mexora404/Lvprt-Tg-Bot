"""
Telegram bot: portrait photo + square driving video -> KlingTeam/LivePortrait on Hugging Face Spaces.
Access control: allow-list; admin commands after /admin + password (ADMIN_PASSWORD_B64).
"""

from __future__ import annotations

import asyncio
import base64
import logging
import os
import re
import secrets
import shutil
import subprocess
import tempfile
import threading
from html import escape
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from dotenv import load_dotenv
from gradio_client import Client, handle_file
from telegram import (
    BotCommand,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    KeyboardButton,
    ReplyKeyboardMarkup,
    Update,
)
from telegram.constants import ParseMode
from telegram.error import BadRequest, TimedOut
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    ConversationHandler,
    MessageHandler,
    filters,
)

import default_driving as dv

from whitelist import is_chat_allowed, load_allowed_ids, save_allowed_ids

load_dotenv()

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

SPACE_ID = "KlingTeam/LivePortrait"
API_ANIMATE = "/gpu_wrapped_execute_video"

MAX_VIDEO_BYTES = 48 * 1024 * 1024
# Telegram full-size photos can be huge / slow; timeout + prefer 2nd-largest size
PHOTO_DOWNLOAD_TIMEOUT_SEC = 90
# Sending result video: PTB default media_write_timeout is 20s — too short for multi-MB uploads
try:
    TELEGRAM_MEDIA_WRITE_TIMEOUT_SEC = int(os.environ.get("TELEGRAM_MEDIA_WRITE_TIMEOUT_SEC", "600"))
except ValueError:
    TELEGRAM_MEDIA_WRITE_TIMEOUT_SEC = 600
try:
    TELEGRAM_READ_TIMEOUT_SEC = int(os.environ.get("TELEGRAM_READ_TIMEOUT_SEC", "300"))
except ValueError:
    TELEGRAM_READ_TIMEOUT_SEC = 300

# White 9:16 export: canvas size + inner box (40% of W and 40% of H max) for centered clip
FIT916_W = 1080
FIT916_H = 1920
FIT916_BOX_FRAC = 0.40
FIT916_SOURCE_KEY = "fit916_source"
CALLBACK_FIT916 = "out:fit916"

WAIT_ADMIN_PW = 0

# Admin: next video message becomes default driving clip (no ConversationHandler)
PENDING_DEFAULT_DV_KEY = "pending_default_driving_upload"

# Reply keyboard labels (must match handler regex)
BTN_MY_ID = "🆔 My ID"
BTN_HELP = "❓ Help"
BTN_HOWTO = "🎬 How to use"
BTN_MAIN = "🏠 Main menu"
# Everyone — visible label so “default video” always shows on keyboard
BTN_DEFAULT_VIDEO_PUBLIC = "🎬 Default video"

# Admin-only row (shown when admin_ok)
BTN_ADMIN_SETVIDEO = "🎬 Set default video"
BTN_ADMIN_LISTUSERS = "📋 Allowed users"
BTN_ADMIN_DEFINFO = "📊 Default video"
BTN_ADMIN_CLEARDEF = "🗑 Clear default video"
BTN_EXIT_ADMIN_KB = "🚪 Exit admin"

_hf_client: Client | None = None
_predict_lock = threading.Lock()


def reply_keyboard_user() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        [
            [KeyboardButton(BTN_MY_ID), KeyboardButton(BTN_HELP)],
            [KeyboardButton(BTN_HOWTO), KeyboardButton(BTN_MAIN)],
            [KeyboardButton(BTN_DEFAULT_VIDEO_PUBLIC)],
        ],
        resize_keyboard=True,
        input_field_placeholder="📷 Portrait · 🎬 Default video…",
        is_persistent=True,
    )


def reply_keyboard_for_context(context: ContextTypes.DEFAULT_TYPE) -> ReplyKeyboardMarkup:
    """Normal user menu, or extra admin row when logged in as admin."""
    if not context.user_data.get("admin_ok"):
        return reply_keyboard_user()
    return ReplyKeyboardMarkup(
        [
            [KeyboardButton(BTN_MY_ID), KeyboardButton(BTN_HELP)],
            [KeyboardButton(BTN_HOWTO), KeyboardButton(BTN_MAIN)],
            [
                KeyboardButton(BTN_ADMIN_SETVIDEO),
                KeyboardButton(BTN_ADMIN_DEFINFO),
            ],
            [
                KeyboardButton(BTN_ADMIN_LISTUSERS),
                KeyboardButton(BTN_ADMIN_CLEARDEF),
            ],
            [KeyboardButton(BTN_EXIT_ADMIN_KB)],
        ],
        resize_keyboard=True,
        input_field_placeholder="📷 Portrait · 🎬 video · ⚙️ admin…",
        is_persistent=True,
    )


def inline_keyboard_denied() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("🆔 My chat ID", callback_data="menu:whoami"),
                InlineKeyboardButton("❓ Help", callback_data="menu:help"),
            ],
        ]
    )


def inline_keyboard_admin_quick() -> InlineKeyboardMarkup:
    """Tap-only admin actions (works alongside reply keyboard; separate message)."""
    return InlineKeyboardMarkup(
        [
            [InlineKeyboardButton("🎬 Set default video", callback_data="admin:setdefault")],
            [
                InlineKeyboardButton("📊 Default status", callback_data="admin:definfo"),
                InlineKeyboardButton("📋 Users", callback_data="admin:listusers"),
            ],
            [InlineKeyboardButton("🗑 Clear default", callback_data="admin:cleardef")],
        ]
    )


def inline_keyboard_home_tap(context: ContextTypes.DEFAULT_TYPE) -> InlineKeyboardMarkup:
    """Tap menu: always show Default video; extra rows when admin."""
    rows: list[list[InlineKeyboardButton]] = [
        [
            InlineKeyboardButton("❓ Help", callback_data="menu:help"),
            InlineKeyboardButton("🆔 My ID", callback_data="menu:whoami"),
        ],
        [InlineKeyboardButton("🎬 Default video (admin)", callback_data="admin:setdefault")],
    ]
    if context.user_data.get("admin_ok"):
        rows.append(
            [
                InlineKeyboardButton("📊 Default status", callback_data="admin:definfo"),
                InlineKeyboardButton("📋 Users", callback_data="admin:listusers"),
            ]
        )
        rows.append([InlineKeyboardButton("⚡ More admin…", callback_data="menu:admin_quick")])
    return InlineKeyboardMarkup(rows)


async def send_inline_menu_follow_up(anchor_message, context: ContextTypes.DEFAULT_TYPE) -> None:
    """After welcome/admin msg that already has reply keyboard — second bubble with tap buttons."""
    await anchor_message.reply_html(
        "⚡ <b>Quick taps</b> — Default video · Help · My ID"
        + (" · status · users" if context.user_data.get("admin_ok") else ""),
        reply_markup=inline_keyboard_home_tap(context),
    )


async def send_inline_admin_follow_up(anchor_message, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Second bubble after admin login — big tap targets for default video etc."""
    await anchor_message.reply_html(
        "👇 <b>Tap — no typing needed:</b>",
        reply_markup=inline_keyboard_admin_quick(),
    )


MENU_TEXT_FILTER = filters.ChatType.PRIVATE & filters.Regex(
    "^(" + "|".join(map(re.escape, [BTN_MY_ID, BTN_HELP, BTN_HOWTO, BTN_MAIN])) + ")$"
)

DEFAULT_VIDEO_TEXT_FILTER = filters.ChatType.PRIVATE & filters.Regex(
    "^" + re.escape(BTN_DEFAULT_VIDEO_PUBLIC) + "$"
)

ADMIN_PANEL_TEXT_FILTER = filters.ChatType.PRIVATE & filters.Regex(
    "^("
    + "|".join(
        map(
            re.escape,
            [
                BTN_ADMIN_SETVIDEO,
                BTN_ADMIN_LISTUSERS,
                BTN_ADMIN_DEFINFO,
                BTN_ADMIN_CLEARDEF,
                BTN_EXIT_ADMIN_KB,
            ],
        )
    )
    + ")$"
)


def get_hf_token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def get_gradio_client() -> Client:
    global _hf_client
    if _hf_client is None:
        kwargs: dict[str, Any] = {}
        t = get_hf_token()
        if t:
            kwargs["token"] = t
        _hf_client = Client(SPACE_ID, **kwargs)
    return _hf_client


def _decode_b64_admin_password() -> str | None:
    b64 = os.environ.get("ADMIN_PASSWORD_B64", "").strip()
    if not b64:
        return None
    try:
        raw = base64.b64decode(b64.encode("ascii")).decode("utf-8")
        return raw.strip("\r\n\t ")
    except Exception:
        logger.error("Invalid ADMIN_PASSWORD_B64")
        return None


def _plain_admin_password_env() -> str | None:
    """Optional plain password in .env — avoids Base64 mistakes."""
    p = os.environ.get("ADMIN_PASSWORD", "").strip()
    return p if p else None


def admin_password_configured() -> bool:
    return bool(_plain_admin_password_env() or _decode_b64_admin_password())


def extract_password_after_admin_or_login(text: str) -> str | None:
    """Parse password from /admin or /login (handles @BotName and @ inside password)."""
    if not text:
        return None
    t = text.strip()
    m = re.match(r"^/(?:admin|login)(?:@[A-Za-z\d_]+)?\s+(.+)$", t, re.DOTALL)
    if not m:
        return None
    return m.group(1).strip("\r\n\t ")


def verify_admin_password(attempt: str) -> bool:
    a = attempt.strip("\r\n\t ").encode("utf-8")
    for src in (_plain_admin_password_env(), _decode_b64_admin_password()):
        if not src:
            continue
        try:
            e = src.encode("utf-8")
            if len(a) != len(e):
                continue
            if secrets.compare_digest(a, e):
                return True
        except Exception:
            continue
    return False


def _verify_admin_start_token(token: str) -> bool:
    """Optional one-shot link: /start &lt;ADMIN_START_SECRET&gt;"""
    secret = os.environ.get("ADMIN_START_SECRET", "").strip()
    if not secret:
        return False
    try:
        t = token.strip().encode("utf-8")
        s = secret.encode("utf-8")
        if len(t) != len(s):
            return False
        return secrets.compare_digest(t, s)
    except Exception:
        return False


def _html_admin_logged_in(cid: int) -> str:
    return (
        "🔐 <b>Admin mode ON</b>\n\n"
        "🎬 <b>Default driving clip:</b> tap <b>Set default video</b> (keyboard ya inline), "
        "phir <b>ek square video</b> bhejo — koi command nahi.\n\n"
        "➕ <code>/addchat &lt;id&gt;</code> — allow user\n"
        "➖ <code>/removechat &lt;id&gt;</code> — remove\n"
        "📋 <code>/listchats</code>\n"
        "📊 <code>/defaultvideo</code> · 🗑 <code>/cleardefaultvideo</code>\n"
        "🚪 <code>/exitadmin</code>\n\n"
        "🆔 <i>Your chat ID:</i> <code>{}</code>\n"
        "<i>Need bot access? /addchat your ID.</i>".format(cid)
    )


def admin_contact_html() -> str:
    c = os.environ.get("ADMIN_CONTACT", "").strip()
    if c:
        return f"\n\n📩 <b>Contact:</b> {escape(c)}"
    return "\n\n📩 Ask the bot owner to add your chat ID."


def can_use_features(context: ContextTypes.DEFAULT_TYPE, chat_id: int) -> bool:
    if context.user_data.get("admin_ok"):
        return True
    return is_chat_allowed(chat_id)


def _unlink_fit916_stash(ud: dict[str, Any]) -> None:
    p = ud.pop(FIT916_SOURCE_KEY, None)
    if p and os.path.isfile(p):
        try:
            os.unlink(p)
        except OSError:
            pass


def clear_media_session(ud: dict[str, Any]) -> None:
    _unlink_fit916_stash(ud)
    ud.pop("portrait_path", None)
    ud.pop("video_path", None)
    ud.pop("busy", None)
    ud.pop(PENDING_DEFAULT_DV_KEY, None)


def _html_whoami(cid: int) -> str:
    return (
        "🆔 <b>Your Telegram chat ID</b>\n\n"
        f"<code>{cid}</code>\n\n"
        "📋 Share this number with an admin so they can run:\n"
        "<code>/addchat {}</code>".format(cid)
    )


def _html_help() -> str:
    quick = ""
    if dv.has_default():
        quick = (
            "\n🎯 <b>Photo-only mode</b> — an admin configured a <b>default driving video</b>. "
            "Send <b>only your portrait</b>; the bot uses that clip automatically.\n"
        )
    return (
        "❓ <b>Help — LivePortrait bot</b>\n\n"
        "🖼 <b>Portrait</b> — one clear face photo (or JPG/PNG file)\n"
        "🎬 <b>Driving video</b> — must be <b>square</b> (e.g. 1080×1080)\n"
        + quick
        + "🔁 <b>Order</b> — photo first or video first, both work\n"
        "⏳ <b>Wait</b> — Hugging Face GPU can take several minutes\n\n"
        "⌨️ <b>Commands</b>\n"
        "/start — Main menu &amp; tips\n"
        "/whoami — Show your chat ID\n"
        "/help — This message\n"
        "/admin or /login — Owner password\n"
        "<i>Or set</i> <code>ADMIN_PASSWORD</code> <i>in .env (plain)</i>\n"
        "<i>Or</i> <code>/start &lt;ADMIN_START_SECRET&gt;</code> <i>if configured</i>\n\n"
        "🔒 <i>Need access?</i>" + admin_contact_html()
    )


def _html_home_welcome() -> str:
    extra = ""
    if dv.has_default():
        extra = "\n\n🎯 <b>Quick mode:</b> admin set a <b>default driving video</b> — you can send <b>photo only</b>."
    return (
        "🎭 <b>LivePortrait</b>\n"
        "┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈\n"
        "✨ <b>Face photo</b> + <b>square driving video</b> → animated clip "
        "(Hugging Face GPU).\n\n"
        "<b>Steps</b>\n"
        "1 · Portrait 📷 or image file\n"
        "2 · Square video (1:1)\n"
        "3 · Wait for render ⏳\n\n"
        "🎬 <b>Default clip</b> — admin can set one; then you only send a portrait. "
        "Button <b>🎬 Default video</b> + <code>/admin</code>.\n\n"
        "<i>Keyboard below · quick taps on the next message.</i>"
        + extra
    )


async def cmd_whoami(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.effective_chat:
        return
    cid = update.effective_chat.id
    kw: dict[str, Any] = {}
    if can_use_features(context, cid):
        kw["reply_markup"] = reply_keyboard_for_context(context)
    await update.effective_message.reply_html(_html_whoami(cid), **kw)


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    kw: dict[str, Any] = {}
    if update.effective_chat and can_use_features(context, update.effective_chat.id):
        kw["reply_markup"] = reply_keyboard_for_context(context)
    await update.effective_message.reply_html(_html_help(), **kw)


async def deny_access_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    cid = update.effective_chat.id
    await update.effective_message.reply_html(
        "🔒 <b>Access restricted</b>\n\n"
        "This bot is <b>invite-only</b>. An admin must approve your chat ID.\n\n"
        f"🆔 <b>Your ID:</b> <code>{cid}</code>\n\n"
        "👇 Tap a button or use /whoami.\n"
        "🔑 Owner: /admin, /login, or <code>ADMIN_PASSWORD</code> in .env"
        + admin_contact_html(),
        reply_markup=inline_keyboard_denied(),
    )


async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.effective_chat:
        return
    cid = update.effective_chat.id

    args = context.args or []
    if args and _verify_admin_start_token(args[0]):
        context.user_data["admin_ok"] = True
        clear_media_session(context.user_data)
        m = await update.effective_message.reply_html(
            "🔐 <b>Admin login OK</b> <i>(start link)</i>\n\n" + _html_admin_logged_in(cid),
            reply_markup=reply_keyboard_for_context(context),
        )
        await send_inline_admin_follow_up(m, context)
        return

    if not can_use_features(context, cid):
        await deny_access_message(update, context)
        return

    clear_media_session(context.user_data)
    home = await update.effective_message.reply_html(
        _html_home_welcome(),
        reply_markup=reply_keyboard_for_context(context),
    )
    await send_inline_menu_follow_up(home, context)


async def send_howto_text(update: Update) -> None:
    await update.effective_message.reply_html(
        "🎬 <b>How to use</b>\n\n"
        "🖼 Portrait + 🎬 <b>square</b> driving video (any order).\n"
        "⚠️ Video must be square — same width &amp; height.\n"
        "⏳ Then wait for Hugging Face processing."
    )


async def menu_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    q = update.callback_query
    if not q:
        return
    data = q.data or ""
    if data == "menu:admin_quick":
        if not context.user_data.get("admin_ok"):
            await q.answer("🔑 Use /admin or /login first.", show_alert=True)
            return
        await q.answer()
        if q.message:
            await send_admin_quick_inline(q.message, context)
        return
    await q.answer()
    if not q.message:
        return
    if data == "menu:whoami" and q.message.chat:
        await q.message.reply_html(_html_whoami(q.message.chat.id))
    elif data == "menu:help":
        await q.message.reply_html(_html_help())


async def output_format_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Inline: generate white 9:16 + 40% centered clip from last raw output."""
    q = update.callback_query
    if not q or (q.data or "") != CALLBACK_FIT916:
        return
    if not q.message:
        await q.answer("No message context.", show_alert=True)
        return
    if not can_use_features(context, q.message.chat_id):
        await q.answer("🔒 Not allowed.", show_alert=True)
        return
    src = context.user_data.get(FIT916_SOURCE_KEY)
    if not src or not os.path.isfile(src):
        await q.answer("Clip missing — run a new render first.", show_alert=True)
        return
    await q.answer()
    fd1, tmp_silent = tempfile.mkstemp(suffix=".mp4", prefix="fit916_s_")
    os.close(fd1)
    fd2, tmp_final = tempfile.mkstemp(suffix=".mp4", prefix="fit916_f_")
    os.close(fd2)
    try:
        await asyncio.to_thread(render_white_916_center_40pct, src, tmp_silent)
        path_to_send = tmp_silent
        if _try_mux_audio_into(tmp_silent, src, tmp_final):
            try:
                os.unlink(tmp_silent)
            except OSError:
                pass
            path_to_send = tmp_final
        else:
            try:
                os.unlink(tmp_final)
            except OSError:
                pass
        if not os.path.isfile(path_to_send) or os.path.getsize(path_to_send) < 32:
            await q.message.reply_text("9:16 export produced an empty file.")
            return
        sz = os.path.getsize(path_to_send)
        if sz > MAX_VIDEO_BYTES:
            await q.message.reply_html(
                f"📦 9:16 export too large for Telegram (~{sz // (1024 * 1024)} MB)."
            )
            return
        cap = "✨ 9:16 white · 40% center"
        with open(path_to_send, "rb") as vf:
            await _reply_video_resilient(q.message, vf, caption=cap)
    except Exception as e:
        logger.exception("9:16 export failed")
        await q.message.reply_text(f"9:16 export failed: {e!s}")
    finally:
        for p in (tmp_silent, tmp_final):
            if p and os.path.isfile(p):
                try:
                    os.unlink(p)
                except OSError:
                    pass


async def send_admin_quick_inline(after_message, context: ContextTypes.DEFAULT_TYPE) -> None:
    await after_message.reply_html(
        "⚡ <b>Quick panel</b> — tap (no typing needed):",
        reply_markup=inline_keyboard_admin_quick(),
    )


async def start_pending_default_video_upload(
    reply_anchor, context: ContextTypes.DEFAULT_TYPE
) -> None:
    context.user_data[PENDING_DEFAULT_DV_KEY] = True
    cancel_kb = InlineKeyboardMarkup(
        [[InlineKeyboardButton("❌ Cancel upload", callback_data="admin:cancel_setdefault")]]
    )
    await reply_anchor.reply_html(
        "🎬 <b>Default driving video</b>\n\n"
        "👉 Ab <b>agle message</b> mein <b>ek hi square video</b> bhejo.\n"
        "📐 Width = height (jaise 1080×1080).\n\n"
        "<i>Cancel:</i> neeche button.",
        reply_markup=cancel_kb,
    )


async def admin_quick_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    q = update.callback_query
    if not q or not q.message:
        return
    data = q.data or ""
    if not data.startswith("admin:"):
        return
    if not context.user_data.get("admin_ok"):
        await q.answer("🔑 Use /admin first", show_alert=True)
        return
    await q.answer()

    if data == "admin:setdefault":
        await start_pending_default_video_upload(q.message, context)
        return
    if data == "admin:cancel_setdefault":
        context.user_data.pop(PENDING_DEFAULT_DV_KEY, None)
        await q.message.reply_html(
            "🚫 Cancelled.",
            reply_markup=reply_keyboard_for_context(context),
        )
        return
    if data == "admin:definfo":
        if not dv.has_default():
            await q.message.reply_html(
                "📭 <b>No default video</b> yet.\n\nTap <b>Set default video</b> above.",
                reply_markup=reply_keyboard_for_context(context),
            )
        else:
            meta = dv.read_meta() or {}
            w, h = meta.get("width"), meta.get("height")
            dim = f"\n📐 <code>{w}×{h}</code>" if w and h else ""
            who = meta.get("set_by_chat_id")
            by = f"\n👤 Set by: <code>{who}</code>" if who else ""
            await q.message.reply_html(
                f"✅ <b>Default video active</b>{dim}{by}",
                reply_markup=reply_keyboard_for_context(context),
            )
        return
    if data == "admin:listusers":
        ids = sorted(load_allowed_ids())
        if not ids:
            await q.message.reply_text("📭 Allow-list is empty.")
        else:
            lines = "\n".join(f"👤 <code>{i}</code>" for i in ids)
            await q.message.reply_html(
                f"📋 <b>Allowed users</b> ({len(ids)})\n\n{lines}",
                reply_markup=reply_keyboard_for_context(context),
            )
        return
    if data == "admin:cleardef":
        dv.clear_files()
        await q.message.reply_text(
            "🗑 Default driving video removed.",
            reply_markup=reply_keyboard_for_context(context),
        )


async def on_default_video_public_button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Everyone sees 🎬 Default video on keyboard; only admin can start upload."""
    if not update.message or not update.effective_chat:
        return
    cid = update.effective_chat.id
    if not can_use_features(context, cid):
        await deny_access_message(update, context)
        return
    if context.user_data.get("admin_ok"):
        await start_pending_default_video_upload(update.message, context)
    else:
        await update.message.reply_html(
            "🎬 <b>Default driving video</b>\n\n"
            "Sirf <b>admin</b> square clip set kar sakta hai (users baad mein sirf photo bhejein).\n\n"
            "👉 Pehle <code>/admin</code> ya <code>/login</code> se login karo, "
            "phir <b>🎬 Default video</b> dubara dabao.",
            reply_markup=reply_keyboard_for_context(context),
        )


async def on_menu_button_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.text:
        return
    if not update.effective_chat:
        return
    cid = update.effective_chat.id
    if not can_use_features(context, cid):
        await deny_access_message(update, context)
        return
    t = update.message.text.strip()
    if t == BTN_MY_ID:
        await update.message.reply_html(_html_whoami(cid))
    elif t == BTN_HELP:
        await update.message.reply_html(_html_help())
    elif t == BTN_HOWTO:
        await send_howto_text(update)
    elif t == BTN_MAIN:
        clear_media_session(context.user_data)
        home = await update.message.reply_html(
            _html_home_welcome(),
            reply_markup=reply_keyboard_for_context(context),
        )
        await send_inline_menu_follow_up(home, context)


async def on_admin_panel_buttons(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Reply-keyboard shortcuts when admin_ok."""
    if not update.message or not update.message.text:
        return
    if not context.user_data.get("admin_ok"):
        return
    t = update.message.text.strip()
    if t == BTN_ADMIN_SETVIDEO:
        await start_pending_default_video_upload(update.message, context)
    elif t == BTN_ADMIN_LISTUSERS:
        await cmd_listchats(update, context)
    elif t == BTN_ADMIN_DEFINFO:
        await cmd_defaultvideo(update, context)
    elif t == BTN_ADMIN_CLEARDEF:
        await cmd_cleardefaultvideo(update, context)
    elif t == BTN_EXIT_ADMIN_KB:
        await cmd_exitadmin(update, context)


async def admin_begin(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    if not update.message:
        return ConversationHandler.END

    text = update.message.text or ""
    pw = extract_password_after_admin_or_login(text)
    if pw is None and context.args:
        pw = " ".join(context.args).strip()

    if pw:
        if verify_admin_password(pw):
            context.user_data["admin_ok"] = True
            m = await update.message.reply_html(
                _html_admin_logged_in(update.effective_chat.id),
                reply_markup=reply_keyboard_for_context(context),
            )
            await send_inline_admin_follow_up(m, context)
            return ConversationHandler.END
        await update.message.reply_text("❌ Wrong password.")
        return ConversationHandler.END

    if not admin_password_configured():
        await update.message.reply_text(
            "⚙️ Set <code>ADMIN_PASSWORD</code> or <code>ADMIN_PASSWORD_B64</code> in .env",
            parse_mode=ParseMode.HTML,
        )
        return ConversationHandler.END

    await update.message.reply_html(
        "🔑 <b>Admin login</b>\n\n"
        "• Send password as the <b>next message</b>\n"
        "• Or: <code>/admin your_password</code> / <code>/login your_password</code>\n"
        "• Or: plain <code>ADMIN_PASSWORD=…</code> in <code>.env</code> (no Base64)\n"
        "• Or: <code>/start &lt;ADMIN_START_SECRET&gt;</code> if set\n\n"
        "/cancel — abort"
    )
    return WAIT_ADMIN_PW


async def admin_receive_password(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    if not update.message or update.message.text is None:
        await update.message.reply_text("Send text password, or /cancel.")
        return WAIT_ADMIN_PW
    if verify_admin_password(update.message.text.strip()):
        context.user_data["admin_ok"] = True
        m = await update.message.reply_html(
            _html_admin_logged_in(update.effective_chat.id),
            reply_markup=reply_keyboard_for_context(context),
        )
        await send_inline_admin_follow_up(m, context)
        return ConversationHandler.END
    await update.message.reply_text("❌ Wrong password. Try again or /cancel.")
    return WAIT_ADMIN_PW


async def admin_cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    if update.message:
        await update.message.reply_text("🚫 Cancelled.")
    return ConversationHandler.END


async def _require_admin_reply(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    if context.user_data.get("admin_ok"):
        return True
    if update.effective_message:
        await update.effective_message.reply_text("🔑 Use /admin with the password first.")
    return False


async def cmd_addchat(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _require_admin_reply(update, context):
        return
    if not context.args:
        await update.effective_message.reply_html(
            "Usage: <code>/addchat &lt;telegram_user_id&gt;</code>\n\nUse /whoami on the user’s phone to get their ID."
        )
        return
    try:
        cid = int(context.args[0].strip())
    except ValueError:
        await update.effective_message.reply_text("ID must be a number.")
        return
    ids = load_allowed_ids()
    ids.add(cid)
    save_allowed_ids(ids)
    await update.effective_message.reply_html(f"✅ Added <code>{cid}</code>\n📊 Total users: <b>{len(ids)}</b>")


async def cmd_removechat(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _require_admin_reply(update, context):
        return
    if not context.args:
        await update.effective_message.reply_html("Usage: <code>/removechat &lt;id&gt;</code>")
        return
    try:
        cid = int(context.args[0].strip())
    except ValueError:
        await update.effective_message.reply_text("ID must be a number.")
        return
    ids = load_allowed_ids()
    ids.discard(cid)
    save_allowed_ids(ids)
    await update.effective_message.reply_html(f"🗑 Removed <code>{cid}</code>\n📊 Total: <b>{len(ids)}</b>")


async def cmd_listchats(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _require_admin_reply(update, context):
        return
    ids = sorted(load_allowed_ids())
    if not ids:
        await update.effective_message.reply_text("📭 Allow-list is empty.")
        return
    lines = "\n".join(f"👤 <code>{i}</code>" for i in ids)
    await update.effective_message.reply_html(f"📋 <b>Allowed users</b> ({len(ids)})\n\n{lines}")


async def cmd_exitadmin(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    context.user_data.pop("admin_ok", None)
    context.user_data.pop(PENDING_DEFAULT_DV_KEY, None)
    if update.effective_message:
        await update.effective_message.reply_text(
            "🚪 Admin session ended.",
            reply_markup=reply_keyboard_for_context(context),
        )


async def cmd_cleardefaultvideo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _require_admin_reply(update, context):
        return
    dv.clear_files()
    await update.effective_message.reply_text("🗑 Default driving video removed. Users must send their own square video again.")


async def cmd_defaultvideo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _require_admin_reply(update, context):
        return
    if not dv.has_default():
        await update.effective_message.reply_html(
            "📭 <b>No default video</b>\n\n"
            "Tap <b>🎬 Set default video</b> on the keyboard or inline menu, then send a square clip.",
            reply_markup=reply_keyboard_for_context(context),
        )
        return
    meta = dv.read_meta() or {}
    w, h = meta.get("width"), meta.get("height")
    dim = f"\n📐 <code>{w}×{h}</code>" if w and h else ""
    who = meta.get("set_by_chat_id")
    by = f"\n👤 Set by chat: <code>{who}</code>" if who else ""
    await update.effective_message.reply_html(
        f"✅ <b>Default driving video</b> is active.{dim}{by}\n\n"
        "Replace: tap <b>Set default video</b> again · Remove: <code>/cleardefaultvideo</code> "
        "or inline <b>Clear default</b>.",
        reply_markup=reply_keyboard_for_context(context),
    )


def video_is_square(path: str) -> tuple[bool, int, int]:
    cap = cv2.VideoCapture(path)
    try:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()
    return (w == h and w > 0), w, h


_VIDEO_FILE_SUFFIXES = (".mp4", ".mov", ".webm", ".mkv", ".m4v", ".avi")


def document_looks_like_video(doc: Any) -> bool:
    """Telegram often sends MP4 as a file with missing mime or application/octet-stream."""
    if not doc:
        return False
    mime = (doc.mime_type or "").lower()
    if mime.startswith("video/"):
        return True
    if mime in ("application/mp4", "application/x-mp4"):
        return True
    fn = (doc.file_name or "").lower()
    if fn and any(fn.endswith(s) for s in _VIDEO_FILE_SUFFIXES):
        return True
    if mime == "application/octet-stream" and fn:
        return any(fn.endswith(s) for s in _VIDEO_FILE_SUFFIXES)
    return False


async def _save_default_driving(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    file_id: str,
    file_size: int | None,
) -> bool:
    """Download Telegram video, ensure square, save as server default for photo-only runs."""
    if file_size and file_size > MAX_VIDEO_BYTES:
        await update.effective_message.reply_text(
            "📦 Too large — max ~48 MB.",
            reply_markup=reply_keyboard_for_context(context),
        )
        return False
    fd, tmp = tempfile.mkstemp(suffix=".mp4", prefix="default_dv_")
    os.close(fd)
    tmp_path = tmp
    try:
        tg_file = await context.bot.get_file(file_id)
        await tg_file.download_to_drive(custom_path=tmp_path)
        ok, w, h = await asyncio.to_thread(video_is_square, tmp_path)
        if not ok:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            await update.effective_message.reply_html(
                f"⚠️ Default must be <b>square</b> (width = height).\n📐 Got <code>{w}×{h}</code>.",
                reply_markup=reply_keyboard_for_context(context),
            )
            return False
        dv.DATA_DIR.mkdir(parents=True, exist_ok=True)
        shutil.move(tmp_path, dv.FILE_PATH)
        tmp_path = None
        if update.effective_chat:
            dv.write_meta(update.effective_chat.id, w, h)
        await update.effective_message.reply_html(
            f"✅ <b>Default driving video saved</b> <code>{w}×{h}</code>\n\n"
            "👥 Users can send <b>only a portrait</b> — this clip is used automatically.",
            reply_markup=reply_keyboard_for_context(context),
        )
        context.user_data.pop(PENDING_DEFAULT_DV_KEY, None)
        logger.info("Default driving video saved (%sx%s)", w, h)
        return True
    except Exception as e:
        logger.exception("save default driving video")
        await update.effective_message.reply_text(
            f"❌ Could not save: {e!s}",
            reply_markup=reply_keyboard_for_context(context),
        )
        return False
    finally:
        if tmp_path and os.path.isfile(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def run_liveportrait(image_path: str, video_path: str) -> tuple[Any, ...]:
    client = get_gradio_client()
    with _predict_lock:
        return client.predict(
            handle_file(image_path),
            {"video": handle_file(video_path)},
            True,
            True,
            True,
            api_name=API_ANIMATE,
        )


def extract_video_path(component: Any) -> str | None:
    if component is None:
        return None
    if isinstance(component, Path):
        return str(component)
    if isinstance(component, str) and (os.path.isabs(component) or component.startswith("http")):
        return component
    if isinstance(component, dict):
        v = component.get("video")
        if isinstance(v, dict) and "path" in v:
            p = v["path"]
            return str(p) if isinstance(p, Path) else p
        if isinstance(v, str):
            return v
        if isinstance(v, Path):
            return str(v)
    return None


def pick_main_output_video_path(result: Any) -> str | None:
    """Space may return 2 clips (main + side-by-side). Never pick concat / preview."""
    if not isinstance(result, (list, tuple)):
        return extract_video_path(result)
    candidates: list[str] = []
    for comp in result:
        p = extract_video_path(comp)
        if p:
            candidates.append(p)
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]

    def looks_concat(path: str) -> bool:
        low = path.lower()
        return (
            "concat" in low
            or "side" in low
            or "grid" in low
            or "compare" in low
            or "stack" in low
        )

    p0, p1 = candidates[0], candidates[1]
    c0, c1 = looks_concat(p0), looks_concat(p1)
    if c0 and not c1:
        return p1
    if c1 and not c0:
        return p0
    return p0


def render_white_916_center_40pct(src_path: str, dst_path: str) -> None:
    """Composite source video on white 9:16 canvas; content scaled to fit inside 40%×40% box, centered."""
    cap = cv2.VideoCapture(src_path)
    if not cap.isOpened():
        raise ValueError("cannot open source video")
    out = None
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
        if fps < 1 or fps > 120:
            fps = 25.0
        W, H = FIT916_W, FIT916_H
        max_w = max(2, int(W * FIT916_BOX_FRAC))
        max_h = max(2, int(H * FIT916_BOX_FRAC))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(dst_path, fourcc, fps, (W, H))
        if not out.isOpened():
            raise RuntimeError("VideoWriter could not open output")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            fh, fw = frame.shape[:2]
            if fw < 1 or fh < 1:
                continue
            scale = min(max_w / fw, max_h / fh)
            nw = max(1, int(round(fw * scale)))
            nh = max(1, int(round(fh * scale)))
            small = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
            canvas = np.full((H, W, 3), 255, dtype=np.uint8)
            x0 = (W - nw) // 2
            y0 = (H - nh) // 2
            canvas[y0 : y0 + nh, x0 : x0 + nw] = small
            out.write(canvas)
    finally:
        if out is not None:
            out.release()
        cap.release()


def _try_mux_audio_into(composite_silent: str, audio_src: str, dst_path: str) -> bool:
    """If ffmpeg is available, mux audio from original clip into composite. Returns True if dst_path ok."""
    if not shutil.which("ffmpeg"):
        return False
    try:
        if os.path.isfile(dst_path):
            os.unlink(dst_path)
    except OSError:
        pass
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                composite_silent,
                "-i",
                audio_src,
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-b:a",
                "128k",
                "-map",
                "0:v:0",
                "-map",
                "1:a:0?",
                "-shortest",
                dst_path,
            ],
            check=True,
            capture_output=True,
            timeout=900,
        )
        return os.path.isfile(dst_path) and os.path.getsize(dst_path) > 0
    except (subprocess.CalledProcessError, OSError, subprocess.TimeoutExpired):
        logger.exception("ffmpeg mux for 9:16 output")
        try:
            if os.path.isfile(dst_path):
                os.unlink(dst_path)
        except OSError:
            pass
        return False


def _pick_photo_file_id(message) -> str:
    """Prefer 2nd-largest size — still sharp for faces, much faster than max resolution."""
    photos = message.photo
    if not photos:
        raise ValueError("no photo")
    if len(photos) >= 2:
        return photos[-2].file_id
    return photos[-1].file_id


async def _download_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> str:
    file_id = _pick_photo_file_id(update.message)
    suffix = ".jpg"
    fd, path = tempfile.mkstemp(suffix=suffix, prefix="portrait_")
    os.close(fd)
    try:
        tg_file = await context.bot.get_file(file_id)
        await tg_file.download_to_drive(custom_path=path)
        return path
    except Exception:
        try:
            os.unlink(path)
        except OSError:
            pass
        raise


async def _download_video_file(update: Update, context: ContextTypes.DEFAULT_TYPE, file_id: str) -> str:
    fd, path = tempfile.mkstemp(suffix=".mp4", prefix="driving_")
    os.close(fd)
    try:
        tg_file = await context.bot.get_file(file_id)
        await tg_file.download_to_drive(custom_path=path)
        return path
    except Exception:
        try:
            os.unlink(path)
        except OSError:
            pass
        raise


def _will_start_liveportrait_after_portrait(context: ContextTypes.DEFAULT_TYPE) -> bool:
    """True if we already have driving video or server default clip."""
    return bool(context.user_data.get("video_path")) or dv.has_default()


async def _edit_msg_html(message, text: str, reply_markup=None) -> None:
    """Edit bot message HTML. Telegram editMessageText only accepts InlineKeyboardMarkup — not ReplyKeyboardMarkup."""
    edit_markup = (
        reply_markup if isinstance(reply_markup, InlineKeyboardMarkup) else None
    )
    try:
        await message.edit_text(text, parse_mode=ParseMode.HTML, reply_markup=edit_markup)
    except BadRequest as e:
        err = ((getattr(e, "message", None) or str(e)) or "").lower()
        if "can't be edited" in err or "message to edit not found" in err:
            logger.warning("edit_text not allowed, sending new message instead: %s", e)
            await message.reply_html(text, reply_markup=reply_markup)
        else:
            raise


async def _safe_edit_or_reply(
    message,
    text: str,
    *,
    parse_mode: str | None = None,
) -> None:
    """Edit a status line; fallback to new message if Telegram rejects edit (e.g. had ReplyKeyboard)."""
    try:
        await message.edit_text(text, parse_mode=parse_mode)
    except BadRequest as e:
        err = ((getattr(e, "message", None) or str(e)) or "").lower()
        if "can't be edited" in err or "message to edit not found" in err:
            logger.warning("status edit failed, sending new message: %s", e)
            if parse_mode == ParseMode.HTML:
                await message.reply_html(text)
            else:
                await message.reply_text(text)
        else:
            raise


async def _reply_video_resilient(message, f, *, caption: str) -> None:
    """Upload video to Telegram; long uploads need high media_write_timeout — retry once on TimedOut."""
    for attempt in range(2):
        try:
            f.seek(0)
            await message.reply_video(video=f, caption=caption)
            return
        except TimedOut:
            if attempt == 0:
                logger.warning("reply_video timed out, retrying once after delay…")
                await asyncio.sleep(3)
                continue
            raise


async def _edit_portrait_ack_saved(
    ack, update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    rk = reply_keyboard_for_context(context)
    if _will_start_liveportrait_after_portrait(context):
        extra = ""
        if dv.has_default() and not context.user_data.get("video_path"):
            extra = "\n🎬 <i>Using admin default driving clip.</i>"
        await _edit_msg_html(
            ack,
            "✅ <b>Portrait saved</b>\n"
            "⚙️ <b>Starting AI render…</b>\n"
            "<i>Progress message next — GPU can take several minutes.</i>"
            + extra,
            reply_markup=rk,
        )
    else:
        await _edit_msg_html(
            ack,
            "✅ <b>Portrait saved</b>\n\n"
            "🎬 <b>Next:</b> send a <b>square</b> driving video\n"
            "<i>Width = height · e.g. 1080×1080</i>",
            reply_markup=rk,
        )


async def on_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.photo:
        return
    if not can_use_features(context, update.effective_chat.id):
        await deny_access_message(update, context)
        return
    if context.user_data.get("busy"):
        await update.message.reply_text("⏳ Already processing…\nUse /start to reset.")
        return

    old = context.user_data.get("portrait_path")
    if old and os.path.isfile(old):
        try:
            os.unlink(old)
        except OSError:
            pass

    rk = reply_keyboard_for_context(context)
    # No reply_markup here — Telegram often returns "Message can't be edited" if we attach
    # ReplyKeyboardMarkup to the same message we later edit.
    ack = await update.message.reply_html(
        "📸 <b>Photo received</b>\n⏳ <i>Downloading…</i> <i>(max ~90s)</i>",
    )
    try:
        path = await asyncio.wait_for(
            _download_photo(update, context),
            timeout=PHOTO_DOWNLOAD_TIMEOUT_SEC,
        )
    except asyncio.TimeoutError:
        logger.warning("Photo download timed out after %ss", PHOTO_DOWNLOAD_TIMEOUT_SEC)
        await _edit_msg_html(
            ack,
            "❌ <b>Download timed out</b>\n\n"
            "Photo file bahut bada / network slow ho sakta hai.\n\n"
            "<b>Try:</b> chhoti photo bhejo, ya Telegram mein <i>Compress</i> on karke bhejo, "
            "ya portrait ko <b>file</b> (JPG) ki tarah bhejo.",
            reply_markup=rk,
        )
        return
    except Exception:
        logger.exception("Photo download failed")
        await _edit_msg_html(
            ack,
            "❌ <b>Could not save photo</b>\n<i>Try again or send as image file.</i>",
            reply_markup=rk,
        )
        raise
    context.user_data["portrait_path"] = path
    await _edit_portrait_ack_saved(ack, update, context)
    await maybe_process(update, context)


async def on_portrait_document(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.document:
        return
    if not can_use_features(context, update.effective_chat.id):
        await deny_access_message(update, context)
        return
    if context.user_data.get("busy"):
        await update.message.reply_text("⏳ Already processing…\nUse /start to reset.")
        return

    old = context.user_data.get("portrait_path")
    if old and os.path.isfile(old):
        try:
            os.unlink(old)
        except OSError:
            pass

    doc = update.message.document
    suffix = ".jpg"
    fn = doc.file_name or ""
    if fn.lower().endswith(".png"):
        suffix = ".png"
    elif fn.lower().endswith(".webp"):
        suffix = ".webp"

    fd, path = tempfile.mkstemp(suffix=suffix, prefix="portrait_")
    os.close(fd)
    rk = reply_keyboard_for_context(context)
    ack = await update.message.reply_html(
        "🖼 <b>Image file received</b>\n⏳ <i>Downloading…</i> <i>(max ~90s)</i>",
    )

    async def _dl_doc() -> None:
        tg_file = await context.bot.get_file(doc.file_id)
        await tg_file.download_to_drive(custom_path=path)

    try:
        await asyncio.wait_for(_dl_doc(), timeout=PHOTO_DOWNLOAD_TIMEOUT_SEC)
    except asyncio.TimeoutError:
        logger.warning("Image file download timed out")
        try:
            os.unlink(path)
        except OSError:
            pass
        await _edit_msg_html(
            ack,
            "❌ <b>Download timed out</b>\n<i>Try a smaller file or check network.</i>",
            reply_markup=rk,
        )
        return
    except Exception:
        logger.exception("Image file download failed")
        try:
            os.unlink(path)
        except OSError:
            pass
        await _edit_msg_html(
            ack,
            "❌ <b>Could not save image</b>\n<i>Try another file.</i>",
            reply_markup=rk,
        )
        raise

    context.user_data["portrait_path"] = path
    await _edit_portrait_ack_saved(ack, update, context)
    await maybe_process(update, context)


async def on_video_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.video:
        return
    if not can_use_features(context, update.effective_chat.id):
        await deny_access_message(update, context)
        return
    vid = update.message.video
    if vid.file_size and vid.file_size > MAX_VIDEO_BYTES:
        await update.message.reply_text("📦 Video too large — max ~48 MB.")
        return
    if context.user_data.get(PENDING_DEFAULT_DV_KEY):
        if not context.user_data.get("admin_ok"):
            context.user_data.pop(PENDING_DEFAULT_DV_KEY, None)
        else:
            await _save_default_driving(update, context, vid.file_id, vid.file_size)
        return
    await _handle_video_file(update, context, vid.file_id)


async def on_video_document(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.document:
        return
    doc = update.message.document
    if not document_looks_like_video(doc):
        return
    if not can_use_features(context, update.effective_chat.id):
        await deny_access_message(update, context)
        return
    if doc.file_size and doc.file_size > MAX_VIDEO_BYTES:
        await update.message.reply_text("📦 Video too large — max ~48 MB.")
        return
    if context.user_data.get(PENDING_DEFAULT_DV_KEY):
        if not context.user_data.get("admin_ok"):
            context.user_data.pop(PENDING_DEFAULT_DV_KEY, None)
        else:
            await _save_default_driving(update, context, doc.file_id, doc.file_size)
        return
    await _handle_video_file(update, context, doc.file_id)


async def on_video_note_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Round video notes: only used when admin is uploading default driving clip."""
    if not update.message or not update.message.video_note:
        return
    if not can_use_features(context, update.effective_chat.id):
        await deny_access_message(update, context)
        return
    vn = update.message.video_note
    if vn.file_size and vn.file_size > MAX_VIDEO_BYTES:
        await update.message.reply_text("📦 Video too large — max ~48 MB.")
        return
    if not (context.user_data.get(PENDING_DEFAULT_DV_KEY) and context.user_data.get("admin_ok")):
        return
    await _save_default_driving(update, context, vn.file_id, vn.file_size)


async def _handle_video_file(update: Update, context: ContextTypes.DEFAULT_TYPE, file_id: str) -> None:
    if context.user_data.get("busy"):
        await update.message.reply_text("⏳ Already processing…\nUse /start to reset.")
        return

    path = await _download_video_file(update, context, file_id)
    ok, w, h = await asyncio.to_thread(video_is_square, path)
    if not ok:
        try:
            os.unlink(path)
        except OSError:
            pass
        await update.message.reply_html(
            "⚠️ Driving video must be <b>square</b> (width = height).\n"
            f"📐 This file: <code>{w}×{h}</code>\n\n"
            "✂️ Crop or export as square, then send again."
        )
        return

    old = context.user_data.get("video_path")
    if old and os.path.isfile(old):
        try:
            os.unlink(old)
        except OSError:
            pass

    context.user_data["video_path"] = path
    if context.user_data.get("portrait_path"):
        await update.message.reply_html(
            f"✅ <b>Square video OK</b> <code>{w}×{h}</code>\n🚀 Starting processing…"
        )
    else:
        await update.message.reply_html(
            f"✅ <b>Square video OK</b> <code>{w}×{h}</code>\n\n"
            "🖼 Next: send a <b>face photo</b> (camera)\n"
            "or an image <b>file</b> (JPG/PNG)."
        )
    await maybe_process(update, context)


async def maybe_process(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    portrait = context.user_data.get("portrait_path")
    driving = context.user_data.get("video_path")
    if portrait and not driving and dv.has_default():
        context.user_data["video_path"] = dv.path_str()
        context.user_data["driving_is_default"] = True
        driving = context.user_data["video_path"]
    if portrait and driving:
        await run_job(update, context)


async def run_job(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    portrait = context.user_data.get("portrait_path")
    driving = context.user_data.get("video_path")
    driving_is_default = context.user_data.get("driving_is_default", False)
    if not portrait or not driving:
        return

    _unlink_fit916_stash(context.user_data)
    context.user_data["busy"] = True
    # No reply_markup — Telegram often rejects editMessageText on messages that had ReplyKeyboardMarkup.
    status = await update.effective_message.reply_html(
        "⏳ <b>Rendering on LivePortrait</b>\n"
        "┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈\n"
        "🖼 Portrait + 🎬 Driving motion → neural render\n"
        "☁️ <b>Hugging Face GPU</b>\n"
        "⏱ <i>Usually 2–15+ min — sit tight.</i>\n"
        "📌 <i>This line updates when your clip is ready.</i>",
    )

    try:
        result = await asyncio.to_thread(run_liveportrait, portrait, driving)
    except Exception as e:
        logger.exception("LivePortrait failed")
        await _safe_edit_or_reply(status, f"❌ Processing failed:\n{e!s}", parse_mode=None)
        return
    finally:
        context.user_data["busy"] = False
        if portrait and os.path.isfile(portrait):
            try:
                os.unlink(portrait)
            except OSError:
                pass
        if not driving_is_default and driving and os.path.isfile(driving):
            try:
                os.unlink(driving)
            except OSError:
                pass
        context.user_data["portrait_path"] = None
        context.user_data["video_path"] = None
        context.user_data.pop("driving_is_default", None)

    paths: list[tuple[str, str]] = []
    main_vp = pick_main_output_video_path(result)
    if main_vp:
        paths.append(("Your clip", main_vp))
    else:
        await _safe_edit_or_reply(status, "❌ No video output from LivePortrait.", parse_mode=None)
        return

    await _safe_edit_or_reply(
        status,
        "✨ <b>Done!</b> Sending your video…",
        parse_mode=ParseMode.HTML,
    )

    for caption, vp in paths:
        if not vp:
            continue
        try:
            if vp.startswith("http"):
                await update.effective_message.reply_html(f"<b>{escape(caption)}</b>\n<code>{vp}</code>")
                continue
            if not os.path.isfile(vp):
                await update.effective_message.reply_text(f"{caption}: file not found on disk.")
                continue
            size = os.path.getsize(vp)
            if size > MAX_VIDEO_BYTES:
                await update.effective_message.reply_html(
                    f"<b>{escape(caption)}</b>\nOutput too large for Telegram (~{size // (1024 * 1024)} MB)."
                )
                continue
            with open(vp, "rb") as f:
                await _reply_video_resilient(
                    update.effective_message,
                    f,
                    caption=f"✨ {caption}",
                )
            try:
                fd, stash = tempfile.mkstemp(suffix=".mp4", prefix="lastout_")
                os.close(fd)
                shutil.copy2(vp, stash)
                context.user_data[FIT916_SOURCE_KEY] = stash
            except Exception:
                logger.exception("stash raw output for 9:16")
            else:
                await update.effective_message.reply_html(
                    "📐 <b>Optional:</b> white <b>9:16</b> background — clip <b>40%</b> centered.\n"
                    "<i>Tap the button below for that version.</i>",
                    reply_markup=InlineKeyboardMarkup(
                        [
                            [
                                InlineKeyboardButton(
                                    "📱 Fit in 9:16 frame",
                                    callback_data=CALLBACK_FIT916,
                                )
                            ]
                        ]
                    ),
                )
        except Exception as e:
            logger.exception("Send video failed")
            await update.effective_message.reply_text(f"{caption}: could not send ({e!s}).")


async def post_init(app: Application) -> None:
    await app.bot.set_my_commands(
        [
            BotCommand("start", "🏠 Main menu"),
            BotCommand("whoami", "🆔 My chat ID"),
            BotCommand("help", "❓ Help & commands"),
            BotCommand("admin", "🔐 Admin login"),
            BotCommand("login", "🔐 Admin login (alias)"),
            BotCommand("defaultvideo", "📊 Default video status"),
            BotCommand("cleardefaultvideo", "🗑 Clear default clip"),
        ]
    )


def main() -> None:
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        raise SystemExit("Set TELEGRAM_BOT_TOKEN in the environment or .env file.")

    admin_conv = ConversationHandler(
        entry_points=[
            CommandHandler("admin", admin_begin),
            CommandHandler("login", admin_begin),
        ],
        states={
            WAIT_ADMIN_PW: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, admin_receive_password),
            ],
        },
        fallbacks=[CommandHandler("cancel", admin_cancel)],
        name="admin_login",
        persistent=False,
    )

    app = (
        Application.builder()
        .token(token)
        .post_init(post_init)
        .media_write_timeout(float(TELEGRAM_MEDIA_WRITE_TIMEOUT_SEC))
        .read_timeout(float(TELEGRAM_READ_TIMEOUT_SEC))
        .build()
    )

    app.add_handler(CallbackQueryHandler(output_format_callback, pattern=r"^out:"))
    app.add_handler(CallbackQueryHandler(menu_callback, pattern=r"^menu:"))
    app.add_handler(CallbackQueryHandler(admin_quick_callback, pattern=r"^admin:"))

    # Group -1 runs before group 0 (admin ConversationHandler)
    for cmd_h in (
        CommandHandler("defaultvideo", cmd_defaultvideo),
        CommandHandler("cleardefaultvideo", cmd_cleardefaultvideo),
        CommandHandler("addchat", cmd_addchat),
        CommandHandler("removechat", cmd_removechat),
        CommandHandler("listchats", cmd_listchats),
        CommandHandler("exitadmin", cmd_exitadmin),
    ):
        app.add_handler(cmd_h, group=-1)

    app.add_handler(CommandHandler("whoami", cmd_whoami))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(admin_conv)
    app.add_handler(MessageHandler(ADMIN_PANEL_TEXT_FILTER, on_admin_panel_buttons))
    app.add_handler(MessageHandler(DEFAULT_VIDEO_TEXT_FILTER, on_default_video_public_button))
    app.add_handler(MessageHandler(MENU_TEXT_FILTER, on_menu_button_text))

    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(MessageHandler(filters.PHOTO, on_photo))
    app.add_handler(MessageHandler(filters.Document.IMAGE, on_portrait_document))
    app.add_handler(MessageHandler(filters.VIDEO, on_video_message))
    app.add_handler(MessageHandler(filters.VIDEO_NOTE, on_video_note_message))
    # MP4 as file often lacks video/* mime; route non-image documents and detect in handler
    app.add_handler(MessageHandler(filters.Document.ALL & ~filters.Document.IMAGE, on_video_document))

    logger.info("Bot starting (long polling)…")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
