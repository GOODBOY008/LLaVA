"""
A controller manages distributed workers.
It sends worker addresses to clients.
"""
import argparse
import json

import requests
import uvicorn
from fastapi import FastAPI, Request

from llava.serve.gradio_web_server import PROMPT_COLOR, PROMPT_STYLE, PROMPT_CATALOG, PROMPT_MATERIAL
from llava.utils import build_logger

logger = build_logger("controller_giga", "controller_giga.log")

app = FastAPI()


def get_worker_addr(model_name):
    controller_url = args.controller_url
    ret = requests.post(controller_url + "/get_worker_address",
                        json={"model": model_name})
    worker_addr = ret.json()["address"]
    logger.info(f"model_name: {model_name}, worker_addr: {worker_addr}")
    return worker_addr


@app.post("/giga_product_feature_detection")
async def giga_product_feature_detection(request: Request):
    system_prompt = ("A chat between a curious human and an artificial intelligence assistant. The assistant gives "
                     "helpful, detailed, and polite answers to the human's questions. USER: <image>\n")
    prompt_arr = [PROMPT_COLOR, PROMPT_STYLE, PROMPT_CATALOG, PROMPT_MATERIAL]
    product_feature_key_arr = ["color", "style", "category", "material"]
    model_name = "llava-v1.5-7b"

    error_msg = None

    worker_addr = get_worker_addr(model_name)
    if worker_addr == "":
        error_msg = "No available worker"
    # Construct prompt and make requests to worker
    result_map = {}
    request_id = ""

    request_json = await request.json()
    title = request_json.get("title", None)
    request_id = request_json.get("request_id", None)
    image = request_json.get("image", None)

    if title is None or request_id is None or image is None:
        error_msg = "title,request_id,image is required"

    if error_msg is None:
        for index, giga_product_prompt in enumerate(prompt_arr):

            giga_product_prompt = giga_product_prompt.replace("{description}", title)
            prompt = system_prompt + giga_product_prompt + " ASSISTANT:"
            payload = {
                "model": model_name,
                "prompt": prompt,
                "temperature": 0.2,
                "top_p": 0.7,
                "max_new_tokens": 512,
                "stop": "</s>",
                "images": [image],
            }

            response = requests.post(worker_addr + "/worker_generate_stream", headers={"User-Agent": "LLaVA Client"},
                                     json=payload, timeout=20)

            for chunk in response.iter_lines(decode_unicode=False, delimiter=b"\0"):
                if chunk:
                    data = json.loads(chunk.decode())
                    if data["error_code"] == 0:
                        product_feature_result = data["text"][len(prompt):].strip()
                        result_map[product_feature_key_arr[index]] = product_feature_result

            result_map[product_feature_key_arr[index]] = json.loads(result_map[product_feature_key_arr[index]])[
                product_feature_key_arr[index]]

        logger.info(f"result_map:{result_map}")
    return {
        "request_id": request_id,
        "data": result_map,
        "error_msg": error_msg,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=21001)
    parser.add_argument("--controller-url", type=str, default= "http://localhost:10000")
    parser.add_argument("--dispatch-method", type=str, choices=[
        "lottery", "shortest_queue"], default="shortest_queue")
    args = parser.parse_args()
    logger.info(f"args: {args}")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
