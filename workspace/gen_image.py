import asyncio
import httpx
from pathlib import Path

async def generate_image():
    api_base = "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation"
    api_key = "sk-bfd7fac70c2e407bb01b1d09dec7a642"
    model_name = "wan2.6-t2i"

    prompt = """A breathtaking mountain landscape at sunset, golden hour lighting,
    snow-capped peaks reflecting on a crystal clear lake, lush green meadows in the foreground,
    dramatic clouds with warm orange and pink hues, photorealistic, cinematic composition"""

    url = f"{api_base}/generation"

    user_content = [{"type": "text", "text": f"Generate a high quality image: {prompt}"}]

    payload = {
        "model": model_name,
        "input": {"messages": [{"role": "user", "content": user_content}]},
        "parameters": {"prompt_extend": True, "watermark": False, "n": 1},
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    print("Generating image, please wait...")

    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.post(url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()

    output = data.get("output") or {}
    choices = output.get("choices") or []

    image_url = None
    if choices:
        message = choices[0].get("message") or {}
        content = message.get("content") or []
        if content and isinstance(content, list):
            image_url = content[0].get("image")

    if image_url:
        print(f"Image generated successfully! Downloading...")

        async with httpx.AsyncClient(timeout=120) as client:
            img_response = await client.get(image_url)
            img_response.raise_for_status()
            img_bytes = img_response.content

        output_path = Path(r"D:\Work\project\nanobot-feishu-specilized\workspace\风景图.png")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(img_bytes)

        print(f"Image saved to: {output_path}")
        return str(output_path)
    else:
        print(f"Generation failed: {data}")
        return None

if __name__ == "__main__":
    result = asyncio.run(generate_image())
