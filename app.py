import streamlit as st
from langgraph.graph import StateGraph
import google.generativeai as genai
import random
import os
import json
from typing import TypedDict, Any
import requests

ACCESS_TOKEN = "AIzaSyDd-JR1M20_vGgCtf0LYCEy1p5YFsDy1ts"

API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={ACCESS_TOKEN}"

def gemini_chat(prompt: str) -> str:
    headers = {
        "Content-Type": "application/json"
    }

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ]
    }

    response = requests.post(API_URL, headers=headers, json=payload)

    if response.status_code == 200:
        result = response.json()
        return result['candidates'][0]['content']['parts'][0]['text']
    else:
        return f"Error: {response.status_code} - {response.text}"

# ----------------------------
# Persistent Session Storage
# ----------------------------

STORAGE_FILE = "sessions.json"

if os.path.exists(STORAGE_FILE):
    with open(STORAGE_FILE, "r") as f:
        SESSIONS = json.load(f)
else:
    SESSIONS = {}

def save_sessions():
    with open(STORAGE_FILE, "w") as f:
        json.dump(SESSIONS, f, indent=2)

# ----------------------------
# Tools
# ----------------------------

def handle_recipe(query: str) -> str:
    prompt = f"""
    You are a cooking assistant. User wants a recipe for:

    {query}

    Provide:
    - Confirmed recipe name
    - Bullet list of ingredients with quantities
    """
    return gemini_chat(prompt)

def product_listing(query: str) -> str:
    products = []
    for _ in range(3):
        brand = random.choice(["India Gate", "Daawat", "Organic Choice"])
        quantity = "1kg"
        price = random.randint(80, 150)
        store = random.choice(["JioMart", "Amazon", "Flipkart"])
        link = "http://example.com/product"

        products.append({
            "ingredient": query,
            "brand": brand,
            "quantity": quantity,
            "price": price,
            "store": store,
            "link": link,
        })

    product_list_str = f"Here are some options for **{query}**:\n\n"
    for p in products:
        product_list_str += (
            f"✅ {p['brand']} {p['ingredient']} {p['quantity']} "
            f"– ₹{p['price']} – {p['store']} – [Link]({p['link']})\n"
        )
    return product_list_str

def add_to_cart(user_id: str, query: str) -> str:
    item = {
        "name": query,
        "price": random.randint(80, 150)
    }
    SESSIONS.setdefault(user_id, []).append(item)
    save_sessions()
    return (
        f"✅ Added **{query}** (₹{item['price']}) to your cart.\n\n"
        f"Would you like to view your cart, remove something, or checkout?"
    )

def view_cart(user_id: str, _: str) -> str:
    cart = SESSIONS.get(user_id, [])
    if not cart:
        return "🛒 Your cart is empty."

    msg = "🛒 **Your Cart:**\n"
    total = 0
    for i, item in enumerate(cart, 1):
        msg += f"{i}. {item['name']} – ₹{item['price']}\n"
        total += item["price"]

    msg += f"\n**Total:** ₹{total}\n"
    msg += "Would you like to remove an item or checkout?"
    return msg

def remove_from_cart(user_id: str, query: str) -> str:
    cart = SESSIONS.get(user_id, [])
    for i, item in enumerate(cart):
        if query.lower() in item["name"].lower():
            removed_item = cart.pop(i)
            save_sessions()
            return f"✅ Removed **{removed_item['name']}** from your cart."

    return f"⚠️ Could not find **{query}** in your cart."

def checkout(user_id: str, _: str) -> str:
    cart = SESSIONS.get(user_id, [])
    if not cart:
        return "Your cart is empty. Add something before checkout!"

    total = sum(item["price"] for item in cart)
    SESSIONS[user_id] = []
    save_sessions()
    return f"✅ Order placed! Total amount: ₹{total}. Thank you for shopping!"

def clear_cart(user_id: str, _: str) -> str:
    SESSIONS[user_id] = []
    save_sessions()
    return "🗑️ Your cart has been cleared."

TOOLS = {
    "recipe_request": handle_recipe,
    "ingredient_query": product_listing,
    "add_to_cart": add_to_cart,
    "view_cart": view_cart,
    "remove_from_cart": remove_from_cart,
    "checkout": checkout,
    "clear_cart": clear_cart,
}

# ----------------------------
# Orchestrator Node
# ----------------------------

def orchestrator_node(state: dict[str, Any]) -> dict[str, Any]:
    user_msg = state["user_message"]
    user_id = state["user_id"]

    classification_prompt = f"""
    Classify this user message into one of:
    - recipe_request
    - ingredient_query
    - add_to_cart
    - view_cart
    - remove_from_cart
    - checkout
    - clear_cart

    ONLY return the keyword, nothing else.

    User message: {user_msg}
    """
    predicted_intent = gemini_chat(classification_prompt).strip().lower()
    print("Predicted intent:", predicted_intent)

    if predicted_intent in TOOLS:
        tool_fn = TOOLS[predicted_intent]
        if predicted_intent in ["add_to_cart", "view_cart", "remove_from_cart", "checkout", "clear_cart"]:
            tool_response = tool_fn(user_id, user_msg)
        else:
            tool_response = tool_fn(user_msg)

        return {
            "assistant_message": tool_response,
            "intent": predicted_intent,
            "user_message": user_msg,
        }
    else:
        response = gemini_chat(
            f"You are a helpful cooking assistant. User said: {user_msg}\nReply politely."
        )
        return {
            "assistant_message": response,
            "intent": "chitchat",
            "user_message": user_msg,
        }

# ----------------------------
# LangGraph Graph
# ----------------------------

class BotState(TypedDict):
    user_id: str
    user_message: str
    assistant_message: str
    intent: str

graph = StateGraph(BotState)
graph.add_node("orchestrator", orchestrator_node)
graph.set_entry_point("orchestrator")
graph.set_finish_point("orchestrator")

app = graph.compile()

# ----------------------------
# Streamlit UI
# ----------------------------

st.set_page_config(page_title="🍳 Recipe AI Assistant", layout="wide")

st.markdown(
    """
    <style>
    .user-msg {
        background-color: #81a366;
        padding: 10px;
        border-radius: 10px;
        margin: 5px 0;
    }
    .bot-msg {
        background-color: #a57e7e;
        padding: 10px;
        border-radius: 10px;
        margin: 5px 0;
    }
    .chat-history {
        max-height: 80vh;
        overflow-y: auto;
        padding-right: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🍳 DishCart - Recipe & Shopping Assistant")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_name = st.text_input("Enter your name:", value="guest")

left_col, right_col = st.columns([2, 3])

# RIGHT column (chat input and actions)
with right_col:
    st.subheader("🤖 Chat with your AI Chef")

    with st.form("chat_form", clear_on_submit=True):
        user_message = st.text_area("Your message:", height=100)
        submitted = st.form_submit_button("Send")

    if submitted and user_message.strip():
        state_in = {
            "user_id": user_name,
            "user_message": user_message,
        }
        with st.spinner("Thinking..."):
            result = app.invoke(state_in)

        st.session_state.chat_history.append(("user", user_message))
        st.session_state.chat_history.append(("bot", result["assistant_message"]))

    st.markdown("---")
    st.subheader("🛒 Cart Actions")

    if st.button("View Cart"):
        response = view_cart(user_name, "")
        st.session_state.chat_history.append(("bot", response))

    if st.button("Checkout"):
        response = checkout(user_name, "")
        st.session_state.chat_history.append(("bot", response))

    if st.button("Clear Cart"):
        response = clear_cart(user_name, "")
        st.session_state.chat_history.append(("bot", response))

    st.write("---")

    remove_item = st.text_input("Remove item from cart:")
    if st.button("Remove"):
        if remove_item:
            response = remove_from_cart(user_name, remove_item)
            st.session_state.chat_history.append(("bot", response))
        else:
            st.warning("Enter item name to remove.")

# LEFT column (history) - placed LAST so it sees the updated history
with left_col:
    st.subheader("💬 Conversation History")
    st.markdown('<div class="chat-history">', unsafe_allow_html=True)

    if st.session_state.chat_history:
        # Group into user-bot pairs
        pairs = []
        temp = []
        for role, msg in st.session_state.chat_history:
            temp.append((role, msg))
            if len(temp) == 2:
                pairs.append(temp)
                temp = []
        if temp:
            pairs.append(temp)

        # Display pairs in descending order
        for pair in reversed(pairs):
            for role, msg in pair:
                if role == "user":
                    st.markdown(
                        f'<div class="user-msg"><strong>You:</strong> {msg}</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f'<div class="bot-msg"><strong>Bot:</strong> {msg}</div>',
                        unsafe_allow_html=True,
                    )
    else:
        st.info("No conversation yet. Start chatting!")

    st.markdown('</div>', unsafe_allow_html=True)
