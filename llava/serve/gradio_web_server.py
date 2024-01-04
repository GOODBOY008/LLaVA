import argparse
import datetime
import json
import os
import time

import gradio as gr
import requests

from fastapi import FastAPI, Request
from llava.conversation import (default_conversation, conv_templates,
                                SeparatorStyle)
from llava.constants import LOGDIR
from llava.utils import (build_logger, server_error_msg,
                         violates_moderation, moderation_msg)
import hashlib

PROMPT_MATERIAL = "Question:  What is the main material used for this product which description is {description} ?\r\n\r\n" \
                  "Candidate material list: [Acrylic,Altay Velvet,Aluminium,Aluminium Alloy,Aluminum,Bamboo," \
                  "Bonded Leather,Bone,Boucle,brass,Bronzing cloth,Bronzing suede,Canvas,Carbon fiber," \
                  "Carbon steel,Ceramic,Chenille,Chrome,Clay,Concrete,Corduroy,Cotton,Cotton Linen,Crystal," \
                  "Down filling,Epoxy resin adhesive,EPU,Eucalyptus,Fabric,faux fur,Fiberglass,Fireclay,Foam," \
                  "genuine leather,Glass,HDPE,Horse hair leather,Iron,Kaolinite,Leather,Linen,Maple,marble,MDF," \
                  "Melamine,Metal,Metal &amp; Wood,Microfiber,Nubuck,Nylon,Nylon Mesh,oxford fabric," \
                  "Palomino Fabric,Particle Board,PC,PET,Pine,Plastic,Plush,Plywood,Polyester,POLYETHYLENE," \
                  "Polypropylene,Polyresin,Polyurethane,POLYVINYL CHLORIDE,PU,PU Leather,PVC,Quartz,Rattan,Resin," \
                  "Rubber,Rubberwood,Sintered Stone,Snowflake Velvet Fabric,Solid Surface,Stainless Steel,Steel," \
                  "Stone,Suede,Tech cloth,technical leather,Textile,Textilene,Upholstered,Velvet,Vinyl,Wicker," \
                  "Solid Wood,Wool,zinc,Olefin,Jute,Viscose,Fiberboard,Paper,Crinkle Oil Paper,Shantung,Parchment," \
                  "Silk,Burlap,Pongee Silk,Taffeta,Dupioni,Grass Cloth,Tissue Shantung,Broadcloth Pleat,Soy wax," \
                  "Recycled paper,Engineered Wood,Water Hyacinth,Porcelain,Woven Rope,Paper Rope,Terrazzo," \
                  "Paper Composite,Copper,Acacia Wood,Magnesium Oxide,Tempered Glass,Microsuede,Fur,Mohair," \
                  "Sheet Metal,Cement,Wood,Sherpa,Polyester Blend,TPU,Engineered Stone,Faux Leather,Cat Scratch " \
                  "Fabric,Teddy,LVL,Fleece,Artificial Marble,Mirror,Synthetic Wood,Mesh,Waterproof Fabric," \
                  "ABS] \r\n\r\n" \
                  "Based on the function, intended use of this object, please analyze the most matching main " \
                  "material and select from the candidate list. You may infer from materials commonly used for " \
                  "similar function objects.\r\n\r\n" \
                  "Answer:\r\n\r\n" \
                  "Return result with follow JSON format {\"material\":XXXX}"

PROMPT_COLOR = "Question: What is the main color of this product which description is {description} ? \r\n\r\n" \
               "Based on the image, please judge the predominant, obvious main color of this product. Reply " \
               "with specific color terms like red,blue, etc. rather than descriptions like " \
               "vibrant or dull." \
               "If there are multiple colors, choose the color with the largest area proportion.\r\n\r\n" \
               "Answer: \r\n\r\n" \
               "Explanation: Briefly explain why you judged this to be the main color, e.g. it occupies the " \
               "largest area percentage.Return with follow JSON format {\"color\":XXXX}"

PROMPT_STYLE = "Question: What is the style of this product?\r\n\r\n" \
               "Candidate style list: [American Design,American Traditional,Antique,Art Deco,Artsy,Beach,Boho," \
               "British,Casual,Chinese,Classic,Coastal,Contemporary,Cute,Desert Lodge,European,Exotic," \
               "Farmhouse,French,French Country,Glam,Grunge,Industrial,Lodge,Luxury,Mid-Century Modern," \
               "Minimalist,Mission,mix match,Modern,Mountain Lodge,Nautical,Ornate Traditional,Pastoral," \
               "Primitivism,Retro,Rustic,Scandinavian,Shabby Chic,Southwestern,Sporty,Traditional,Transitional," \
               "Tropical,Ultra-Modern,Victorian,Vintage,Wild Style]\r\n\r\n" \
               "Answer:  \r\n\r\n" \
               "Based on the image, please judge the most likely style for this product. Return with follow " \
               "JSON format {\"style\":XXXX}"

PROMPT_CATALOG = "Question: What category does this product belong to which description is {description} ?\r\n\r\n " \
                 "Candidate categories: Pens & Hutches, Trees & Condos, Grooming, Pet Beds & Furniture, Kids Bikes & " \
                 "Riding Toys, Rockers, Swing Sets, Outdoor Sports, Kids Kitchen Playsets, Kids Slides, Soft " \
                 "Play, Beds/Frames & Bases, Benches/Stools, Nightstands, Makeup Vanities, Jewelry " \
                 "Storage, Daybeds, Bedroom Sets, Bedroom " \
                 "Storage, Dressers/Chests/Wardrobes, Sofas, Tables, Chairs/Accent Seating, Bean Bag Chairs/Lazy Sofa " \
                 "Chair, TV/Entertainment Furniture, Indoor Fireplaces, Storage Benches, Trash " \
                 "Cans, Sectionals, Loveseats, Recliners/Massage Chairs, Rocking Chairs, Cabinets, Ottomans, Coat " \
                 "Racks, Display/Shelving/Etageres, Office Chairs, File Cabinets/Storage Cabinets, Desks/Work " \
                 "Surfaces, Safes, Seating for Dining, Kitchen Islands & Carts, Dining and Kitchen Sets, Dining " \
                 "Tables, Servers/Sideboards/Buffets, Table Benches, Youth/Kids/Baby Furniture, Game " \
                 "Tables, Seating/Chairs, Massage Tables, Parts, Mobile Scooters, Winches, Car Roof Tents, Hand Trucks & " \
                 "Dollies, Automotive Interior Coolers, Car Jacks, Vibration Platforms, Elliptical Trainers, Inversion " \
                 "Equipment, Rowers, Trampolines, Step Machines, Treadmills, Exercise Bikes, Fishing Kayaks, Weight " \
                 "Benches, Water sports, Weight Racks, Other Exercise Equipment, Outdoor Bikes, Skateboards, Table " \
                 "Tennis Tables, Gym Mats, Basketball Hoops, Golf Bag Carts, Golf Sets, Inflatable Paddle Boards, Kick " \
                 "Scooters, Soccer Tables, Pool Tables, Christmas Trees, Bathroom Mirrors, Full Length " \
                 "Mirrors, Clocks, Accessories, Rugs, Wall Art, Flowers & Plants, Mattresses, Blankets & Pillows, Bedding " \
                 "Sets, Sheets & Pillowcases, Mattress Protectors, Comforters, Curtains, Privacy Screens, Wine " \
                 "Cellars, Ice Makers, Refrigerators, Kitchen Sinks, Dish Drying Racks, Spice Racks, Kitchen " \
                 "Faucets, Food Sanitizer, Wine Racks, Espresso Machines, Accessories, Dishwashers, Kitchen Range " \
                 "Hoods, Cooktops, Dryers, Bathtubs, Shower Doors, Vanity Sinks, Toilets & Bidets, Saunas, Bathroom Sink " \
                 "Faucets, Freestanding Tub Faucets, Bathroom Storage, Bathroom Accessories, Heat Press Machines, Air " \
                 "Compressors, Display Freezers & Refrigerators, Beauty & Personal Care, Water Fountains, Patio " \
                 "Seating, Patio Furniture Sets, Outdoor Tables, Fences, Umbrellas & Shades, Carports, Outdoor " \
                 "Heating, Grills and Smokers, Garden Carts, Garden Pots & Planters, Garden Arch & Trellis, Pools, Weed " \
                 "Barrier Fabric, Outdoor Generators & Portable Power, Accessories, Vacuums & Floor Cleaning " \
                 "Machines, Air Conditioners, Fans, Home Cleaning, Dryers, Dehumidifiers, Ovens, Office Electronics, Soft " \
                 "Case, Hard Case, Backpacks, Other, Audio Accessories & Musical Instruments, Ventilation Fans, Wire " \
                 "Fencing, Ladders, Tool Boxes, Tool Cabinets, Snow Removal Tools, Skid Steer Auger Drives & " \
                 "Bits, Power & Pneumatic Tools, Blowers, Wall Treatment & Supplies, Doors & Door Hardware, Dining " \
                 "Room Lighting, Bathroom Lighting, Lighting/Lamps, Living Room Lighting, Bedroom Lighting, Outdoor " \
                 "Lighting, Lighting Accessories \r\n\r\n" \
                 "Please judge the category this product image belongs to, and select the result from the " \
                 "candidate categories above.\r\n\r\n" \
                 "Answer:  \r\n\r\n" \
                 "Explanation: Briefly explain the reasoning behind your judged category, e.g. the product's " \
                 "function, shape, materials etc. that are characteristic for that category.Return with follow " \
                 "JSON format {\"category\":XXXX}"

logger = build_logger("gradio_web_server", "gradio_web_server.log")

headers = {"User-Agent": "LLaVA Client"}

no_change_btn = gr.Button.update()
enable_btn = gr.Button.update(interactive=True)
disable_btn = gr.Button.update(interactive=False)

priority = {
    "vicuna-13b": "aaaaaaa",
    "koala-13b": "aaaaaab",
}

def get_conv_log_filename():
    t = datetime.datetime.now()
    name = os.path.join(LOGDIR, f"{t.year}-{t.month:02d}-{t.day:02d}-conv.json")
    return name


def get_model_list():
    ret = requests.post(args.controller_url + "/refresh_all_workers")
    assert ret.status_code == 200
    ret = requests.post(args.controller_url + "/list_models")
    models = ret.json()["models"]
    models.sort(key=lambda x: priority.get(x, x))
    logger.info(f"Models: {models}")
    return models


get_window_url_params = """
function() {
    const params = new URLSearchParams(window.location.search);
    url_params = Object.fromEntries(params);
    console.log(url_params);
    return url_params;
    }
"""


def load_demo(url_params, request: gr.Request):
    logger.info(f"load_demo. ip: {request.client.host}. params: {url_params}")

    dropdown_update = gr.Dropdown.update(visible=True)
    if "model" in url_params:
        model = url_params["model"]
        if model in models:
            dropdown_update = gr.Dropdown.update(
                value=model, visible=True)

    state = default_conversation.copy()
    return state, dropdown_update


def load_demo_refresh_model_list(request: gr.Request):
    logger.info(f"load_demo. ip: {request.client.host}")
    models = get_model_list()
    state = default_conversation.copy()
    dropdown_update = gr.Dropdown.update(
        choices=models,
        value=models[0] if len(models) > 0 else ""
    )
    return state, dropdown_update


def vote_last_response(state, vote_type, model_selector, request: gr.Request):
    with open(get_conv_log_filename(), "a") as fout:
        data = {
            "tstamp": round(time.time(), 4),
            "type": vote_type,
            "model": model_selector,
            "state": state.dict(),
            "ip": request.client.host,
        }
        fout.write(json.dumps(data) + "\n")


def upvote_last_response(state, model_selector, request: gr.Request):
    logger.info(f"upvote. ip: {request.client.host}")
    vote_last_response(state, "upvote", model_selector, request)
    return ("",) + (disable_btn,) * 3


def downvote_last_response(state, model_selector, request: gr.Request):
    logger.info(f"downvote. ip: {request.client.host}")
    vote_last_response(state, "downvote", model_selector, request)
    return ("",) + (disable_btn,) * 3


def flag_last_response(state, model_selector, request: gr.Request):
    logger.info(f"flag. ip: {request.client.host}")
    vote_last_response(state, "flag", model_selector, request)
    return ("",) + (disable_btn,) * 3


def regenerate(state, image_process_mode, request: gr.Request):
    logger.info(f"regenerate. ip: {request.client.host}")
    state.messages[-1][-1] = None
    prev_human_msg = state.messages[-2]
    if type(prev_human_msg[1]) in (tuple, list):
        prev_human_msg[1] = (*prev_human_msg[1][:2], image_process_mode)
    state.skip_next = False
    return (state, state.to_gradio_chatbot(), "", None) + (disable_btn,) * 5


def clear_history(request: gr.Request):
    logger.info(f"clear_history. ip: {request.client.host}")
    state = default_conversation.copy()
    return (state, state.to_gradio_chatbot(), "", None) + (disable_btn,) * 5


def add_text(state, text, image, image_process_mode, request: gr.Request):
    logger.info(f"add_text. ip: {request.client.host}. len: {len(text)}")
    if len(text) <= 0 and image is None:
        state.skip_next = True
        return (state, state.to_gradio_chatbot(), "", None) + (no_change_btn,) * 5
    if args.moderate:
        flagged = violates_moderation(text)
        if flagged:
            state.skip_next = True
            return (state, state.to_gradio_chatbot(), moderation_msg, None) + (
                no_change_btn,) * 5
    # text = text[:1536]  # Hard cut-off
    if image is not None:
        # text = text[:1200]  # Hard cut-off for images
        if '<image>' not in text:
            # text = '<Image><image></Image>' + text
            text = text + '\n<image>'
        text = (text, image, image_process_mode)
        if len(state.get_images(return_pil=True)) > 0:
            state = default_conversation.copy()
    state.append_message(state.roles[0], text)
    state.append_message(state.roles[1], None)
    state.skip_next = False
    return (state, state.to_gradio_chatbot(), "", None) + (disable_btn,) * 5


def http_bot(state, model_selector, temperature, top_p, max_new_tokens, request: gr.Request):
    logger.info(f"http_bot. ip: {request.client.host}")
    start_tstamp = time.time()
    model_name = model_selector

    if state.skip_next:
        # This generate call is skipped due to invalid inputs
        yield (state, state.to_gradio_chatbot()) + (no_change_btn,) * 5
        return

    if len(state.messages) == state.offset + 2:
        # First round of conversation
        if "llava" in model_name.lower():
            if 'llama-2' in model_name.lower():
                template_name = "llava_llama_2"
            elif "mistral" in model_name.lower() or "mixtral" in model_name.lower():
                if 'orca' in model_name.lower():
                    template_name = "mistral_orca"
                elif 'hermes' in model_name.lower():
                    template_name = "chatml_direct"
                else:
                    template_name = "mistral_instruct"
            elif 'llava-v1.6-34b' in model_name.lower():
                template_name = "chatml_direct"
            elif "v1" in model_name.lower():
                if 'mmtag' in model_name.lower():
                    template_name = "v1_mmtag"
                elif 'plain' in model_name.lower() and 'finetune' not in model_name.lower():
                    template_name = "v1_mmtag"
                else:
                    template_name = "llava_v1"
            elif "mpt" in model_name.lower():
                template_name = "mpt"
            else:
                if 'mmtag' in model_name.lower():
                    template_name = "v0_mmtag"
                elif 'plain' in model_name.lower() and 'finetune' not in model_name.lower():
                    template_name = "v0_mmtag"
                else:
                    template_name = "llava_v0"
        elif "mpt" in model_name:
            template_name = "mpt_text"
        elif "llama-2" in model_name:
            template_name = "llama_2"
        else:
            template_name = "vicuna_v1"
        new_state = conv_templates[template_name].copy()
        new_state.append_message(new_state.roles[0], state.messages[-2][1])
        new_state.append_message(new_state.roles[1], None)
        state = new_state

    # Query worker address
    worker_addr = get_worker_addr(model_name)

    # No available worker
    if worker_addr == "":
        state.messages[-1][-1] = server_error_msg
        yield (state, state.to_gradio_chatbot(), disable_btn, disable_btn, disable_btn, enable_btn, enable_btn)
        return

    # Construct prompt
    prompt = state.get_prompt()

    all_images = state.get_images(return_pil=True)
    all_image_hash = [hashlib.md5(image.tobytes()).hexdigest() for image in all_images]
    for image, hash in zip(all_images, all_image_hash):
        t = datetime.datetime.now()
        filename = os.path.join(LOGDIR, "serve_images", f"{t.year}-{t.month:02d}-{t.day:02d}", f"{hash}.jpg")
        if not os.path.isfile(filename):
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            image.save(filename)

    # Make requests
    pload = {
        "model": model_name,
        "prompt": prompt,
        "temperature": float(temperature),
        "top_p": float(top_p),
        "max_new_tokens": min(int(max_new_tokens), 1536),
        "stop": state.sep if state.sep_style in [SeparatorStyle.SINGLE, SeparatorStyle.MPT] else state.sep2,
        "images": f'List of {len(state.get_images())} images: {all_image_hash}',
    }

    pload['images'] = state.get_images()

    logger.info(f"==== request ====\n{pload}")

    state.messages[-1][-1] = "▌"
    yield (state, state.to_gradio_chatbot()) + (disable_btn,) * 5

    try:
        # Stream output
        response = requests.post(worker_addr + "/worker_generate_stream",
                                 headers=headers, json=pload, stream=True, timeout=10)
        for chunk in response.iter_lines(decode_unicode=False, delimiter=b"\0"):
            if chunk:
                data = json.loads(chunk.decode())
                if data["error_code"] == 0:
                    output = data["text"][len(prompt):].strip()
                    state.messages[-1][-1] = output + "▌"
                    yield (state, state.to_gradio_chatbot()) + (disable_btn,) * 5
                else:
                    output = data["text"] + f" (error_code: {data['error_code']})"
                    state.messages[-1][-1] = output
                    yield (state, state.to_gradio_chatbot()) + (
                        disable_btn, disable_btn, disable_btn, enable_btn, enable_btn)
                    return
                time.sleep(0.03)
    except requests.exceptions.RequestException as e:
        state.messages[-1][-1] = server_error_msg
        yield (state, state.to_gradio_chatbot()) + (disable_btn, disable_btn, disable_btn, enable_btn, enable_btn)
        return

    state.messages[-1][-1] = state.messages[-1][-1][:-1]
    yield (state, state.to_gradio_chatbot()) + (enable_btn,) * 5

    finish_tstamp = time.time()
    logger.info(f"{output}")

    with open(get_conv_log_filename(), "a") as fout:
        data = {
            "tstamp": round(finish_tstamp, 4),
            "type": "chat",
            "model": model_name,
            "start": round(start_tstamp, 4),
            "finish": round(finish_tstamp, 4),
            "state": state.dict(),
            "images": all_image_hash,
            "ip": request.client.host,
            "output": output,
        }
        fout.write(json.dumps(data) + "\n")


def get_worker_addr(model_name):
    controller_url = args.controller_url
    ret = requests.post(controller_url + "/get_worker_address",
                        json={"model": model_name})
    worker_addr = ret.json()["address"]
    logger.info(f"model_name: {model_name}, worker_addr: {worker_addr}")
    return worker_addr


title_markdown = ("""
""")

tos_markdown = ("""
""")

learn_more_markdown = ("""
""")

block_css = """

#buttons button {
    min-width: min(120px,100%);
}

"""


def build_demo(embed_mode):
    textbox = gr.Textbox(show_label=False, placeholder="Enter text and press ENTER", container=False)
    with gr.Blocks(title="LLaVA", theme=gr.themes.Default(), css=block_css) as demo:
        state = gr.State()

        if not embed_mode:
            gr.Markdown(title_markdown)

        with gr.Row():
            with gr.Column(scale=3):
                with gr.Row(elem_id="model_selector_row"):
                    model_selector = gr.Dropdown(
                        choices=models,
                        value=models[0] if len(models) > 0 else "",
                        interactive=True,
                        show_label=False,
                        container=False)

                imagebox = gr.Image(type="pil")
                image_process_mode = gr.Radio(
                    ["Crop", "Resize", "Pad", "Default"],
                    value="Default",
                    label="Preprocess for non-square image", visible=False)

                cur_dir = os.path.dirname(os.path.abspath(__file__))
                gr.Examples(examples=[
                    [f"{cur_dir}/examples/sofa.jpg",
                     PROMPT_CATALOG],
                    [f"{cur_dir}/examples/sofa-1.jpg",
                     PROMPT_STYLE],
                    [f"{cur_dir}/examples/sofa-2.jpg",
                     PROMPT_COLOR],
                    [f"{cur_dir}/examples/chair.jpg",
                     PROMPT_MATERIAL],
                ], inputs=[imagebox, textbox])

                with gr.Accordion("Parameters", open=False) as parameter_row:
                    temperature = gr.Slider(minimum=0.0, maximum=1.0, value=0.2, step=0.1, interactive=True,
                                            label="Temperature", )
                    top_p = gr.Slider(minimum=0.0, maximum=1.0, value=0.7, step=0.1, interactive=True, label="Top P", )
                    max_output_tokens = gr.Slider(minimum=0, maximum=1024, value=512, step=64, interactive=True,
                                                  label="Max output tokens", )

            with gr.Column(scale=8):
                chatbot = gr.Chatbot(elem_id="chatbot", label="LLaVA Chatbot", height=550)
                with gr.Row():
                    with gr.Column(scale=8):
                        textbox.render()
                    with gr.Column(scale=1, min_width=50):
                        submit_btn = gr.Button(value="Send", variant="primary")
                with gr.Row(elem_id="buttons") as button_row:
                    upvote_btn = gr.Button(value="👍  Upvote", interactive=False)
                    downvote_btn = gr.Button(value="👎  Downvote", interactive=False)
                    flag_btn = gr.Button(value="⚠️  Flag", interactive=False)
                    # stop_btn = gr.Button(value="⏹️  Stop Generation", interactive=False)
                    regenerate_btn = gr.Button(value="🔄  Regenerate", interactive=False)
                    clear_btn = gr.Button(value="🗑️  Clear", interactive=False)

        if not embed_mode:
            gr.Markdown(tos_markdown)
            gr.Markdown(learn_more_markdown)
        url_params = gr.JSON(visible=False)

        # Register listeners
        btn_list = [upvote_btn, downvote_btn, flag_btn, regenerate_btn, clear_btn]
        upvote_btn.click(
            upvote_last_response,
            [state, model_selector],
            [textbox, upvote_btn, downvote_btn, flag_btn],
            queue=False
        )
        downvote_btn.click(
            downvote_last_response,
            [state, model_selector],
            [textbox, upvote_btn, downvote_btn, flag_btn],
            queue=False
        )
        flag_btn.click(
            flag_last_response,
            [state, model_selector],
            [textbox, upvote_btn, downvote_btn, flag_btn],
            queue=False
        )

        regenerate_btn.click(
            regenerate,
            [state, image_process_mode],
            [state, chatbot, textbox, imagebox] + btn_list,
            queue=False
        ).then(
            http_bot,
            [state, model_selector, temperature, top_p, max_output_tokens],
            [state, chatbot] + btn_list
        )

        clear_btn.click(
            clear_history,
            None,
            [state, chatbot, textbox, imagebox] + btn_list,
            queue=False
        )

        textbox.submit(
            add_text,
            [state, textbox, imagebox, image_process_mode],
            [state, chatbot, textbox, imagebox] + btn_list,
            queue=False
        ).then(
            http_bot,
            [state, model_selector, temperature, top_p, max_output_tokens],
            [state, chatbot] + btn_list
        )

        submit_btn.click(
            add_text,
            [state, textbox, imagebox, image_process_mode],
            [state, chatbot, textbox, imagebox] + btn_list,
            queue=False
        ).then(
            http_bot,
            [state, model_selector, temperature, top_p, max_output_tokens],
            [state, chatbot] + btn_list
        )

        if args.model_list_mode == "once":
            demo.load(
                load_demo,
                [url_params],
                [state, model_selector],
                _js=get_window_url_params,
                queue=False
            )
        elif args.model_list_mode == "reload":
            demo.load(
                load_demo_refresh_model_list,
                None,
                [state, model_selector],
                queue=False
            )
        else:
            raise ValueError(f"Unknown model list mode: {args.model_list_mode}")

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int)
    parser.add_argument("--controller-url", type=str, default="http://localhost:21001")
    parser.add_argument("--concurrency-count", type=int, default=10)
    parser.add_argument("--model-list-mode", type=str, default="once",
                        choices=["once", "reload"])
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--moderate", action="store_true")
    parser.add_argument("--embed", action="store_true")
    args = parser.parse_args()
    logger.info(f"args: {args}")

    models = get_model_list()

    logger.info(args)
    demo = build_demo(args.embed)
    demo.queue(
        concurrency_count=args.concurrency_count,
        api_open=False
    ).launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share
    )
