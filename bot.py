"""
魅物AI Telegram Bot — ComfyUI 适配版
适用于 CloudStudio / 本地 ComfyUI (端口 8188)
"""

import nest_asyncio
nest_asyncio.apply()

# ============================================================
#  ★ 配置区 — 只需修改这里
# ============================================================
BOT_TOKEN = "8661379207:AAHepLe9Bu4rWCusSs7ebOd1Yn6aeVgLRZA"

# ComfyUI 地址（不要加末尾斜杠）
COMFYUI_BASE = "https://606fb0af23994bc59ea27c05ea50bbc1--8188.ap-shanghai2.cloudstudio.club"

# 聚合支付地址
PAYMENT_API_URL = "https://payment.fw.com//now"

# ============================================================
#  ComfyUI API 封装
# ============================================================
import requests, json, uuid, time, asyncio, logging, io, base64
from typing import Optional

logger = logging.getLogger(__name__)

# ComfyUI 文生图 workflow 模板（使用最基础的 KSampler 流程）
def build_txt2img_workflow(prompt: str, seed: int = -1) -> dict:
    if seed == -1:
        seed = int(time.time()) % 2147483647
    return {
        "3": {
            "class_type": "KSampler",
            "inputs": {
                "seed": seed, "steps": 20, "cfg": 7,
                "sampler_name": "euler", "scheduler": "normal",
                "denoise": 1.0,
                "model":       ["4", 0],
                "positive":    ["6", 0],
                "negative":    ["7", 0],
                "latent_image":["5", 0],
            }
        },
        "4": {"class_type": "CheckpointLoaderSimple",
              "inputs": {"ckpt_name": "v1-5-pruned-emaonly.ckpt"}},
        "5": {"class_type": "EmptyLatentImage",
              "inputs": {"width": 512, "height": 512, "batch_size": 1}},
        "6": {"class_type": "CLIPTextEncode",
              "inputs": {"text": prompt, "clip": ["4", 1]}},
        "7": {"class_type": "CLIPTextEncode",
              "inputs": {"text": "ugly, blurry, low quality", "clip": ["4", 1]}},
        "8": {"class_type": "VAEDecode",
              "inputs": {"samples": ["3", 0], "vae": ["4", 2]}},
        "9": {"class_type": "SaveImage",
              "inputs": {"filename_prefix": "mw_ai", "images": ["8", 0]}},
    }

# ComfyUI 图生图 workflow（含 LoadImage 节点）
def build_img2img_workflow(prompt: str, image_name: str, seed: int = -1) -> dict:
    if seed == -1:
        seed = int(time.time()) % 2147483647
    return {
        "1": {"class_type": "LoadImage",
              "inputs": {"image": image_name, "upload": "image"}},
        "2": {"class_type": "VAEEncode",
              "inputs": {"pixels": ["1", 0], "vae": ["4", 2]}},
        "3": {"class_type": "KSampler",
              "inputs": {
                  "seed": seed, "steps": 20, "cfg": 7,
                  "sampler_name": "euler", "scheduler": "normal",
                  "denoise": 0.75,
                  "model":       ["4", 0],
                  "positive":    ["6", 0],
                  "negative":    ["7", 0],
                  "latent_image":["2", 0],
              }},
        "4": {"class_type": "CheckpointLoaderSimple",
              "inputs": {"ckpt_name": "v1-5-pruned-emaonly.ckpt"}},
        "6": {"class_type": "CLIPTextEncode",
              "inputs": {"text": prompt, "clip": ["4", 1]}},
        "7": {"class_type": "CLIPTextEncode",
              "inputs": {"text": "ugly, blurry, low quality", "clip": ["4", 1]}},
        "8": {"class_type": "VAEDecode",
              "inputs": {"samples": ["3", 0], "vae": ["4", 2]}},
        "9": {"class_type": "SaveImage",
              "inputs": {"filename_prefix": "mw_ai", "images": ["8", 0]}},
    }


def get_available_checkpoint() -> Optional[str]:
    """从 ComfyUI 获取第一个可用的模型文件名"""
    try:
        r = requests.get(f"{COMFYUI_BASE}/object_info/CheckpointLoaderSimple",
                         timeout=10)
        if r.status_code == 200:
            info = r.json()
            ckpts = info.get("CheckpointLoaderSimple", {}) \
                        .get("input", {}).get("required", {}) \
                        .get("ckpt_name", [[]])[0]
            if ckpts:
                logger.info(f"可用模型: {ckpts}")
                return ckpts[0]
    except Exception as e:
        logger.warning(f"获取模型列表失败: {e}")
    return None


async def upload_image_to_comfyui(img_bytes: bytes) -> Optional[str]:
    """上传参考图到 ComfyUI，返回图片名"""
    loop = asyncio.get_event_loop()
    def _upload():
        files = {"image": ("ref.png", img_bytes, "image/png"),
                 "overwrite": (None, "true")}
        r = requests.post(f"{COMFYUI_BASE}/upload/image",
                          files=files, timeout=30)
        return r
    try:
        r = await loop.run_in_executor(None, _upload)
        if r.status_code == 200:
            name = r.json().get("name")
            logger.info(f"图片上传成功: {name}")
            return name
    except Exception as e:
        logger.error(f"上传图片失败: {e}")
    return None


async def comfyui_generate(workflow: dict, timeout: int = 120) -> Optional[bytes]:
    """
    提交 workflow → 轮询队列 → 获取生成结果图片
    返回 PNG bytes，失败返回 None
    """
    loop = asyncio.get_event_loop()
    client_id = str(uuid.uuid4())

    # 1. 提交任务
    def _submit():
        return requests.post(
            f"{COMFYUI_BASE}/prompt",
            json={"prompt": workflow, "client_id": client_id},
            timeout=30
        )
    try:
        r = await loop.run_in_executor(None, _submit)
        if r.status_code != 200:
            logger.error(f"提交任务失败: {r.status_code} {r.text[:300]}")
            return None
        prompt_id = r.json().get("prompt_id")
        logger.info(f"任务已提交, prompt_id={prompt_id}")
    except Exception as e:
        logger.error(f"提交任务异常: {e}")
        return None

    # 2. 轮询历史，等待完成
    deadline = time.time() + timeout
    while time.time() < deadline:
        await asyncio.sleep(3)
        try:
            h = await loop.run_in_executor(
                None,
                lambda: requests.get(f"{COMFYUI_BASE}/history/{prompt_id}", timeout=10)
            )
            if h.status_code != 200:
                continue
            history = h.json()
            if prompt_id not in history:
                continue
            outputs = history[prompt_id].get("outputs", {})
            # 找到第一张输出图片
            for node_id, node_out in outputs.items():
                images = node_out.get("images", [])
                if images:
                    img_info = images[0]
                    filename  = img_info["filename"]
                    subfolder = img_info.get("subfolder", "")
                    img_type  = img_info.get("type", "output")
                    # 3. 下载图片
                    params = {"filename": filename,
                              "subfolder": subfolder,
                              "type": img_type}
                    img_r = await loop.run_in_executor(
                        None,
                        lambda: requests.get(
                            f"{COMFYUI_BASE}/view",
                            params=params, timeout=30
                        )
                    )
                    if img_r.status_code == 200:
                        logger.info(f"图片下载成功, size={len(img_r.content)}")
                        return img_r.content
        except Exception as e:
            logger.warning(f"轮询异常: {e}")

    logger.error(f"生图超时（>{timeout}s）")
    return None


async def call_image_api(prompt: str,
                         ref_bytes: Optional[bytes] = None) -> Optional[bytes]:
    """
    对外统一接口：
    - 自动检测 ComfyUI 是否可用
    - 自动选择可用模型
    - 支持文生图 / 图生图
    """
    # 检查 ComfyUI 是否在线
    try:
        loop = asyncio.get_event_loop()
        ping = await loop.run_in_executor(
            None,
            lambda: requests.get(f"{COMFYUI_BASE}/system_stats", timeout=8)
        )
        if ping.status_code != 200:
            logger.error(f"ComfyUI 不在线: {ping.status_code}")
            return None
    except Exception as e:
        logger.error(f"ComfyUI 连接失败: {e}")
        return None

    # 获取可用模型
    ckpt = get_available_checkpoint() or "v1-5-pruned-emaonly.ckpt"

    if ref_bytes:
        # 图生图
        img_name = await upload_image_to_comfyui(ref_bytes)
        if not img_name:
            return None
        wf = build_img2img_workflow(prompt, img_name)
    else:
        # 文生图
        wf = build_txt2img_workflow(prompt)

    # 更新 workflow 中的模型名
    for node in wf.values():
        if node.get("class_type") == "CheckpointLoaderSimple":
            node["inputs"]["ckpt_name"] = ckpt

    return await comfyui_generate(wf)


# ============================================================
#  以下为 Bot 主体（与修复版相同，直接复用）
# ============================================================
from io import BytesIO
from typing import Dict, Any
from PIL import Image, ImageDraw, ImageFont

from telegram import (
    Update, InlineKeyboardButton, InlineKeyboardMarkup, BotCommand,
)
from telegram.ext import (
    Application, CommandHandler, MessageHandler,
    CallbackQueryHandler, ConversationHandler,
    filters, ContextTypes,
)
from telegram.constants import ParseMode

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
)

DISCLAIMER = """⚠️使用条款和免责声明：
➡️该模式内容请务必在私密环境下独自享用！
➡️本功能根据用户的输入生成图像，但不对用户使用它创建的任何特定图像负责。
➡️用户在使用此功能时必须对内容和行为承担全部责任。
❌禁止传播可能对个人或组织造成伤害的图像。
❌严禁非法侵权：严禁对未成年人、公众人物进行肖像篡改或色情化生成。
❌严禁违法传播：严禁将生成内容用于商业牟利、勒索、诈骗或恶意诋毁。
❌严禁非法采集：用户必须确保上传的图片已获得权利人明确授权。
⚠️警告：利用AI技术制作并传播淫秽色情信息或侵犯他人名誉权在多个司法管辖区属违法犯罪行为，请合规使用。"""

PACKAGES = {
    "platinum": {"name": "白金套餐",  "images": 50,  "price": 100},
    "diamond":  {"name": "钻石套餐",  "images": 110, "price": 200},
    "supreme":  {"name": "至尊套餐",  "images": 280, "price": 500},
}
users: Dict[int, Dict[str, Any]] = {}
pending_payments: Dict[str, Dict] = {}
(S_TXT2IMG, S_IMG2IMG_UPLOAD, S_IMG2IMG_PROMPT, S_CLONE_TOKEN) = range(4)


def get_user(uid):
    if uid not in users:
        users[uid] = {
            "name": "", "username": "",
            "images_left": 3, "images_used": 0,
            "invited_by": None, "invited_count": 0,
            "packages": [], "bots": [],
            "joined_ts": int(time.time()),
        }
    return users[uid]


def main_keyboard():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🎨 文生图",   callback_data="txt2img"),
         InlineKeyboardButton("🖼 图生图",   callback_data="img2img")],
        [InlineKeyboardButton("💎 套餐购买", callback_data="buy_package"),
         InlineKeyboardButton("👤 用户中心", callback_data="user_center")],
        [InlineKeyboardButton("🎁 邀请激励", callback_data="invite"),
         InlineKeyboardButton("🤖 机器人克隆", callback_data="clone_bot")],
        [InlineKeyboardButton("🚀 加入AI共创，祝你日入过万！", callback_data="ai_partner")],
    ])


def gen_welcome_image():
    try:
        w, h = 800, 400
        img = Image.new("RGB", (w, h))
        draw = ImageDraw.Draw(img)
        for y in range(h):
            draw.line([(0,y),(w,y)],
                      fill=(int(20+y*30/h), int(5+y*15/h), int(40+y*100/h)))
        for cx,cy,rad,col in [(120,90,70,(180,80,220)),
                               (680,310,90,(60,100,255)),
                               (400,40,35,(255,190,50))]:
            draw.ellipse([cx-rad,cy-rad,cx+rad,cy+rad], fill=col)
        try:
            fb = ImageFont.truetype(
                "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc", 56)
            fs = ImageFont.truetype(
                "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 30)
        except Exception:
            fb = fs = ImageFont.load_default()
        draw.text((402,h//2-55),"✨ 魅物AI ✨",fill=(0,0,0),font=fb,anchor="mm")
        draw.text((400,h//2-57),"✨ 魅物AI ✨",fill=(255,220,80),font=fb,anchor="mm")
        draw.text((402,h//2+22),"AI生图神器，满足你的一切幻想",fill=(0,0,0),font=fs,anchor="mm")
        draw.text((400,h//2+20),"AI生图神器，满足你的一切幻想",fill=(210,190,255),font=fs,anchor="mm")
        buf = BytesIO(); img.save(buf,"JPEG",quality=92); buf.seek(0)
        return buf
    except Exception:
        return None


async def finish_image(update, context, img_bytes, prompt):
    uid = update.effective_user.id
    u = get_user(uid)
    msg = update.effective_message
    if img_bytes:
        u["images_left"] -= 1
        u["images_used"] += 1
        await msg.reply_photo(
            photo=BytesIO(img_bytes),
            caption=f"✅ 图片生成成功！\n📝 提示词：{prompt[:120]}"
        )
    else:
        await msg.reply_text(
            "❌ 生图失败。请确认：\n"
            f"1. CloudStudio 会话仍在运行\n"
            f"2. ComfyUI 已正常启动（无报错）\n"
            f"3. 模型文件已放入 models/checkpoints/\n"
            f"4. API 地址：{COMFYUI_BASE}"
        )
    await msg.reply_text(DISCLAIMER)
    await msg.reply_text(
        f"📊 剩余次数：{u['images_left']} 次",
        reply_markup=main_keyboard()
    )


async def complete_payment(uid, pkg_key, context):
    pkg = PACKAGES[pkg_key]
    u = get_user(uid)
    u["images_left"] += pkg["images"]
    u["packages"].append({"pkg": pkg["name"], "images": pkg["images"],
                           "price": pkg["price"], "ts": int(time.time())})
    await context.bot.send_message(
        uid,
        f"🎉 支付成功！\n✅ {pkg['name']}\n🖼 到账：{pkg['images']} 次\n📊 剩余：{u['images_left']} 次",
        reply_markup=main_keyboard()
    )


async def cmd_start(update, context):
    user = update.effective_user
    uid  = user.id
    u = get_user(uid)
    u["name"] = user.full_name
    u["username"] = user.username or ""
    if context.args:
        arg = context.args[0]
        if arg.startswith("ref_"):
            try:
                inviter_id = int(arg[4:])
                if inviter_id != uid and u["invited_by"] is None:
                    u["invited_by"] = inviter_id
                    inv = get_user(inviter_id)
                    inv["invited_count"] += 1
                    inv["images_left"]   += 1
                    u["images_left"]     += 1
                    await context.bot.send_message(
                        inviter_id,
                        f"🎉 好友 {user.full_name} 加入了魅物AI！\n你获得1次奖励，剩余：{inv['images_left']} 次"
                    )
            except Exception:
                pass
    img = gen_welcome_image()
    cap = "🌟 *欢迎使用魅物AI！*\n\n_AI生图神器，满足你的一切幻想_\n\n🎁 新用户赠送 *3次* 免费次数！"
    if img:
        await update.message.reply_photo(photo=img, caption=cap,
                                         parse_mode=ParseMode.MARKDOWN,
                                         reply_markup=main_keyboard())
    else:
        await update.message.reply_text(cap, parse_mode=ParseMode.MARKDOWN,
                                        reply_markup=main_keyboard())


async def cmd_cancel(update, context):
    context.user_data.clear()
    await update.message.reply_text("✅ 已取消", reply_markup=main_keyboard())
    return ConversationHandler.END


async def txt2img_entry(update, context):
    query = update.callback_query; await query.answer()
    u = get_user(query.from_user.id)
    if u["images_left"] < 1:
        await query.message.reply_text("⚠️ 次数不足，请购买套餐",
            reply_markup=InlineKeyboardMarkup([[
                InlineKeyboardButton("💎 购买套餐", callback_data="buy_package")]]))
        return ConversationHandler.END
    await query.message.reply_text(
        "✏️ *文生图*\n在对话框输入提示词即可，提示词越丰富图片越带劲！\n\n/cancel 取消",
        parse_mode=ParseMode.MARKDOWN)
    return S_TXT2IMG


async def txt2img_do(update, context):
    uid = update.effective_user.id
    u = get_user(uid)
    prompt = update.message.text.strip()
    if u["images_left"] < 1:
        await update.message.reply_text("⚠️ 次数不足，请购买套餐",
                                        reply_markup=main_keyboard())
        return ConversationHandler.END
    wait = await update.message.reply_text("⏳ 生成中，请稍候（约20-60秒）…")
    img_bytes = await call_image_api(prompt)
    try: await wait.delete()
    except Exception: pass
    await finish_image(update, context, img_bytes, prompt)
    return ConversationHandler.END


async def img2img_entry(update, context):
    query = update.callback_query; await query.answer()
    u = get_user(query.from_user.id)
    if u["images_left"] < 1:
        await query.message.reply_text("⚠️ 次数不足，请购买套餐",
            reply_markup=InlineKeyboardMarkup([[
                InlineKeyboardButton("💎 购买套餐", callback_data="buy_package")]]))
        return ConversationHandler.END
    await query.message.reply_text("🖼 *图生图*\n请先发送一张参考图👇\n\n/cancel 取消",
                                   parse_mode=ParseMode.MARKDOWN)
    return S_IMG2IMG_UPLOAD


async def img2img_got_photo(update, context):
    photo = update.message.photo[-1]
    f = await photo.get_file()
    context.user_data["i2i_ref"] = bytes(await f.download_as_bytearray())
    await update.message.reply_text("✅ 参考图收到！\n\n请输入提示词，越丰富越带劲！\n\n/cancel 取消")
    return S_IMG2IMG_PROMPT


async def img2img_do(update, context):
    uid = update.effective_user.id
    u = get_user(uid)
    prompt   = update.message.text.strip()
    ref_data = context.user_data.pop("i2i_ref", None)
    if not ref_data:
        await update.message.reply_text("❌ 未找到参考图，请重新开始",
                                        reply_markup=main_keyboard())
        return ConversationHandler.END
    if u["images_left"] < 1:
        await update.message.reply_text("⚠️ 次数不足，请购买套餐",
                                        reply_markup=main_keyboard())
        return ConversationHandler.END
    wait = await update.message.reply_text("⏳ 生成中，请稍候（约20-60秒）…")
    img_bytes = await call_image_api(prompt, ref_data)
    try: await wait.delete()
    except Exception: pass
    await finish_image(update, context, img_bytes, prompt)
    return ConversationHandler.END


async def clone_entry(update, context):
    query = update.callback_query; await query.answer()
    await query.message.reply_text(
        "🤖 *克隆机器人*\n\n"
        "📝 步骤：\n1️⃣ @BotFather → /newbot\n2️⃣ 获取 Token\n3️⃣ 发送 Token 到这里\n\n"
        "格式：`123456789:ABCdefGHIjklMNOpqrsTUVwxyz`\n\n/cancel 取消",
        parse_mode=ParseMode.MARKDOWN)
    return S_CLONE_TOKEN


async def clone_got_token(update, context):
    token = update.message.text.strip()
    uid   = update.effective_user.id
    if ":" not in token or len(token) < 30:
        await update.message.reply_text("❌ Token 格式不对，请重试。\n/cancel 取消")
        return S_CLONE_TOKEN
    wait = await update.message.reply_text("⏳ 验证中…")
    loop = asyncio.get_event_loop()
    try:
        r = await loop.run_in_executor(
            None, lambda: requests.get(
                f"https://api.telegram.org/bot{token}/getMe", timeout=15))
        data = r.json()
        if not data.get("ok"):
            await wait.edit_text("❌ Token 无效，请检查后重试。")
            return S_CLONE_TOKEN
        info  = data["result"]
        uname = info.get("username","")
        get_user(uid)["bots"].append(token)
        loop.create_task(launch_cloned_bot(token, uid))
        await wait.edit_text(
            f"✅ *克隆成功！*\n🤖 @{uname}\n👑 你已成为该Bot团长！",
            parse_mode=ParseMode.MARKDOWN, reply_markup=main_keyboard())
    except Exception as e:
        await wait.edit_text(f"❌ 失败：{str(e)[:100]}")
        return S_CLONE_TOKEN
    return ConversationHandler.END


async def launch_cloned_bot(token, owner_id):
    try:
        app = Application.builder().token(token).build()
        setup_all_handlers(app)
        await app.initialize(); await app.start()
        await app.updater.start_polling(drop_pending_updates=True)
        logger.info(f"克隆Bot已启动 owner={owner_id}")
        while True: await asyncio.sleep(3600)
    except Exception as e:
        logger.error(f"克隆Bot异常 owner={owner_id}: {e}")


async def cb_main_menu(update, context):
    query = update.callback_query; await query.answer()
    await query.message.reply_text("请选择功能👇", reply_markup=main_keyboard())


async def cb_buy_package(update, context):
    query = update.callback_query; await query.answer()
    kb = [
        [InlineKeyboardButton("🥈 白金套餐  50张  ¥100", callback_data="pkg_platinum")],
        [InlineKeyboardButton("💎 钻石套餐 110张  ¥200", callback_data="pkg_diamond")],
        [InlineKeyboardButton("👑 至尊套餐 280张  ¥500", callback_data="pkg_supreme")],
        [InlineKeyboardButton("🔙 返回", callback_data="main_menu")],
    ]
    await query.message.reply_text("💎 *套餐购买*\n请选择套餐：",
                                   parse_mode=ParseMode.MARKDOWN,
                                   reply_markup=InlineKeyboardMarkup(kb))


async def cb_select_pkg(update, context):
    query = update.callback_query; await query.answer()
    pkg_key = query.data[4:]
    pkg = PACKAGES.get(pkg_key)
    if not pkg: return
    context.user_data["pending_pkg"] = pkg_key
    kb = [
        [InlineKeyboardButton("💚 微信支付",   callback_data=f"pay_wechat_{pkg_key}")],
        [InlineKeyboardButton("💙 支付宝支付", callback_data=f"pay_alipay_{pkg_key}")],
        [InlineKeyboardButton("🔙 返回套餐",   callback_data="buy_package")],
    ]
    await query.message.reply_text(
        f"✅ *{pkg['name']}*\n🖼 {pkg['images']}张  💰 ¥{pkg['price']}\n\n请选择支付方式：",
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=InlineKeyboardMarkup(kb))


async def cb_pay(update, context):
    query = update.callback_query; await query.answer()
    parts = query.data.split("_", 2)
    method, pkg_key = parts[1], parts[2]
    pkg = PACKAGES.get(pkg_key)
    if not pkg: return
    uid = query.from_user.id
    order_id = str(uuid.uuid4())
    method_cn = "微信支付" if method == "wechat" else "支付宝支付"
    wait = await query.message.reply_text(f"⏳ 正在生成 {method_cn} 链接…")
    loop = asyncio.get_event_loop()
    try:
        r = await loop.run_in_executor(
            None, lambda: requests.post(
                PAYMENT_API_URL,
                json={"order_id": order_id, "amount": pkg["price"],
                      "method": method, "user_id": str(uid),
                      "description": "魅物AI套餐购买"},
                timeout=30))
        pay_url = None
        if r.status_code == 200:
            d = r.json()
            pay_url = d.get("pay_url") or d.get("qr_url") or d.get("url")
    except Exception:
        pay_url = None

    if pay_url:
        await wait.edit_text(
            f"✅ *{method_cn}*\n\n{pay_url}\n\n⏱ 有效期15分钟，支付成功后自动到账。",
            parse_mode=ParseMode.MARKDOWN)
        asyncio.get_event_loop().call_later(
            5, lambda: asyncio.ensure_future(complete_payment(uid, pkg_key, context)))
    else:
        await wait.edit_text(
            f"⚠️ 支付API暂未配置，调试模式直接充值 {pkg['images']} 次。")
        await complete_payment(uid, pkg_key, context)


async def cb_user_center(update, context):
    query = update.callback_query; await query.answer()
    uid = query.from_user.id
    u = get_user(uid)
    last = u["packages"][-1] if u["packages"] else None
    last_txt = f"📦 {last['pkg']} · {last['images']}张 · ¥{last['price']}" if last else "📦 暂无购买记录"
    join = time.strftime("%Y-%m-%d", time.localtime(u["joined_ts"]))
    await query.message.reply_text(
        f"👤 *用户中心*\n\n🆔 `{uid}`\n👤 {u['name']}\n📅 {join}\n"
        f"━━━━━━━━━━━━━━━━\n🖼 剩余：*{u['images_left']}* 次\n📊 已用：{u['images_used']} 次\n"
        f"━━━━━━━━━━━━━━━━\n{last_txt}\n"
        f"━━━━━━━━━━━━━━━━\n🤖 已克隆Bot：{len(u['bots'])} 个\n👥 已邀请：{u['invited_count']} 人",
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 返回", callback_data="main_menu")]]))


async def cb_invite(update, context):
    query = update.callback_query; await query.answer()
    uid = query.from_user.id
    u = get_user(uid)
    me = await context.bot.get_me()
    link = f"https://t.me/{me.username}?start=ref_{uid}"
    share = f"https://t.me/share/url?url={link}&text=快来加入魅物AI，免费获得图片生成次数！"
    await query.message.reply_text(
        f"🎁 *邀请激励*\n\n每邀请1位好友：\n👤 你+1次  🆕 好友+1次\n"
        f"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n🔗 你的邀请链接：\n`{link}`\n"
        f"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n📈 已邀请：*{u['invited_count']}* 位",
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=InlineKeyboardMarkup([
            [InlineKeyboardButton("📤 分享邀请链接", url=share)],
            [InlineKeyboardButton("🔙 返回", callback_data="main_menu")]]))


async def cb_ai_partner(update, context):
    query = update.callback_query; await query.answer()
    await query.message.reply_text(
        "🚀 【AI合伙人代理 · 收益规则】\n\n"
        "━━━━ ① 拉新奖励 ━━━━\n你+1次，新人+1次，折合0.3U\n\n"
        "━━━━ ② 三级返佣 ━━━━\n一级20% · 二级10% · 三级5%\n\n"
        "━━━━ ③ 团长体系 ━━━━\n一级40% · 二级10% · 三级5%\n\n"
        "🎯 机器人进的群越多，赚的钱越多！",
        reply_markup=InlineKeyboardMarkup([
            [InlineKeyboardButton("🎁 邀请激励",   callback_data="invite"),
             InlineKeyboardButton("🤖 机器人克隆", callback_data="clone_bot")],
            [InlineKeyboardButton("🔙 返回",       callback_data="main_menu")]]))


async def error_handler(update, context):
    logger.error("全局异常: %s", context.error, exc_info=context.error)
    if isinstance(update, Update) and update.effective_message:
        await update.effective_message.reply_text(
            "⚠️ 内部错误，请发送 /start 重新开始。")


def setup_all_handlers(app):
    app.add_handler(ConversationHandler(
        entry_points=[CallbackQueryHandler(txt2img_entry, pattern="^txt2img$")],
        states={S_TXT2IMG: [MessageHandler(filters.TEXT & ~filters.COMMAND, txt2img_do)]},
        fallbacks=[CommandHandler("cancel", cmd_cancel)], per_message=False))
    app.add_handler(ConversationHandler(
        entry_points=[CallbackQueryHandler(img2img_entry, pattern="^img2img$")],
        states={
            S_IMG2IMG_UPLOAD: [MessageHandler(filters.PHOTO, img2img_got_photo)],
            S_IMG2IMG_PROMPT: [MessageHandler(filters.TEXT & ~filters.COMMAND, img2img_do)],
        },
        fallbacks=[CommandHandler("cancel", cmd_cancel)], per_message=False))
    app.add_handler(ConversationHandler(
        entry_points=[CallbackQueryHandler(clone_entry, pattern="^clone_bot$")],
        states={S_CLONE_TOKEN: [MessageHandler(filters.TEXT & ~filters.COMMAND, clone_got_token)]},
        fallbacks=[CommandHandler("cancel", cmd_cancel)], per_message=False))
    app.add_handler(CommandHandler("start",  cmd_start))
    app.add_handler(CommandHandler("cancel", cmd_cancel))
    app.add_handler(CallbackQueryHandler(cb_main_menu,   pattern="^main_menu$"))
    app.add_handler(CallbackQueryHandler(cb_buy_package, pattern="^buy_package$"))
    app.add_handler(CallbackQueryHandler(cb_select_pkg,  pattern="^pkg_(platinum|diamond|supreme)$"))
    app.add_handler(CallbackQueryHandler(cb_pay,         pattern="^pay_(wechat|alipay)_"))
    app.add_handler(CallbackQueryHandler(cb_user_center, pattern="^user_center$"))
    app.add_handler(CallbackQueryHandler(cb_invite,      pattern="^invite$"))
    app.add_handler(CallbackQueryHandler(cb_ai_partner,  pattern="^ai_partner$"))
    app.add_error_handler(error_handler)


def main():
    app = Application.builder().token(BOT_TOKEN).build()
    setup_all_handlers(app)

    async def on_startup(a):
        await a.bot.set_my_commands([
            BotCommand("start",  "🏠 主菜单"),
            BotCommand("cancel", "❌ 取消操作"),
        ])
        me = await a.bot.get_me()
        print(f"✅ 魅物AI Bot 启动：@{me.username}")
        print(f"📡 ComfyUI = {COMFYUI_BASE}")
        print(f"💳 支付API = {PAYMENT_API_URL}")
        print("=" * 50)

    app.post_init = on_startup
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
