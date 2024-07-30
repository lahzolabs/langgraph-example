import os
import shutil
import sqlite3
import pandas as pd
import requests
from typing import TypedDict, Annotated, Sequence, Literal
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import tools_condition
from langchain_anthropic import ChatAnthropic, Anthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig

from langgraph.graph.message import AnyMessage, add_messages
from dotenv import load_dotenv
load_dotenv()

REPHRASER = "rephraser"


# db_url = "https://storage.googleapis.com/benchmarks-artifacts/travel-db/travel2.sqlite"
# local_file = "travel2.sqlite"
# # The backup lets us restart for each tutorial section
# backup_file = "travel2.backup.sqlite"
# overwrite = False
# if overwrite or not os.path.exists(local_file):
#     response = requests.get(db_url)
#     response.raise_for_status()  # Ensure the request was successful
#     with open(local_file, "wb") as f:
#         f.write(response.content)
#     # Backup - we will use this to "reset" our DB in each section
#     shutil.copy(local_file, backup_file)
# # Convert the flights to present time for our tutorial
# conn = sqlite3.connect(local_file)
# cursor = conn.cursor()

# tables = pd.read_sql(
#     "SELECT name FROM sqlite_master WHERE type='table';", conn
# ).name.tolist()
# tdf = {}
# for t in tables:
#     tdf[t] = pd.read_sql(f"SELECT * from {t}", conn)

# example_time = pd.to_datetime(
#     tdf["flights"]["actual_departure"].replace("\\N", pd.NaT)
# ).max()
# current_time = pd.to_datetime("now").tz_localize(example_time.tz)
# time_diff = current_time - example_time

# tdf["bookings"]["book_date"] = (
#     pd.to_datetime(tdf["bookings"]["book_date"].replace("\\N", pd.NaT), utc=True)
#     + time_diff
# )

# datetime_columns = [
#     "scheduled_departure",
#     "scheduled_arrival",
#     "actual_departure",
#     "actual_arrival",
# ]
# for column in datetime_columns:
#     tdf["flights"][column] = (
#         pd.to_datetime(tdf["flights"][column].replace("\\N", pd.NaT)) + time_diff
#     )

# for table_name, df in tdf.items():
#     df.to_sql(table_name, conn, if_exists="replace", index=False)
# del df
# del tdf
# conn.commit()
# conn.close()

# db = local_file  # We'll be using this local file as our DB in this tutorial


import re

import numpy as np
import openai
from langchain_core.tools import tool

response = requests.get(
    "https://storage.googleapis.com/benchmarks-artifacts/travel-db/swiss_faq.md"
)
response.raise_for_status()
faq_text = response.text

docs = [{"page_content": txt} for txt in re.split(r"(?=\n##)", faq_text)]


class VectorStoreRetriever:
    def __init__(self, docs: list, vectors: list, oai_client):
        self._arr = np.array(vectors)
        self._docs = docs
        self._client = oai_client

    @classmethod
    def from_docs(cls, docs, oai_client):
        embeddings = oai_client.embeddings.create(
            model="text-embedding-3-small", input=[doc["page_content"] for doc in docs]
        )
        vectors = [emb.embedding for emb in embeddings.data]
        return cls(docs, vectors, oai_client)

    def query(self, query: str, k: int = 5) -> list[dict]:
        embed = self._client.embeddings.create(
            model="text-embedding-3-small", input=[query]
        )
        # "@" is just a matrix multiplication in python
        scores = np.array(embed.data[0].embedding) @ self._arr.T
        top_k_idx = np.argpartition(scores, -k)[-k:]
        top_k_idx_sorted = top_k_idx[np.argsort(-scores[top_k_idx])]
        return [
            {**self._docs[idx], "similarity": scores[idx]} for idx in top_k_idx_sorted
        ]


retriever = VectorStoreRetriever.from_docs(docs, openai.Client())



import sqlite3
from datetime import date, datetime
from typing import Optional

import pytz
from langchain_core.runnables import ensure_config

# Sales Assistant Tools
#dealership_by_name, dealership_locator, send_contanct_info
@tool
def dealership_by_name() -> str:
    """
    Search for a dealership by name.
    """

    return "The dealership is Tucker, GA. It opens at 6am and closes at 6pm."

@tool
def dealership_locator(zip_code: str) -> str:
    """
    Locate the nearest dealership to the user's zip code.
    """
    return "The nearest dealership is in Tucker, GA"

@tool
def send_contanct_info(name: str, phone: str=None, email: str=None) -> str:
    """
    Send the contact information to the dealership.
    """
    return "The contact information has been sent to the dealership."

# Inventory Assistant Tools
# check_rv_inventory_by_dealership, check_rv_inventory_by_zipcode
@tool
def check_rv_inventory_by_dealership(dealership_name: str, rv_model: str="") -> str:
    """
    Check the RV inventory by dealership name.
    """
    return "The RV inventory at the dealership is as follows: ..."

@tool
def check_rv_inventory_by_zipcode(zip_code: str, rv_model: str="") -> str:
    """
    Check the RV inventory by zip code.
    """
    return "The RV inventory at the dealership is as follows: ..."

# Scheduling Assistant Tools
# schedule_visit_by_inventory, schedule_visit_by_location
@tool
def schedule_visit_by_inventory(inventory_url: str) -> str:
    """
    Schedule a visit based on the inventory item.
    """
    return "The visit has been scheduled for the specified RV unit."

@tool
def schedule_visit_by_location(dealership_name: str) -> str:
    """
    Schedule a visit based on the customer's zip code.
    """
    return "The visit has been scheduled at the specified dealership."

    

from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableLambda

from langgraph.prebuilt import ToolNode


def handle_tool_error(state) -> dict:
    error = state.get("error")
    tool_calls = state["messages"][-1].tool_calls
    return {
        "messages": [
            ToolMessage(
                content=f"Error: {repr(error)}\n please fix your mistakes.",
                tool_call_id=tc["id"],
            )
            for tc in tool_calls
        ]
    }


def create_tool_node_with_fallback(tools: list) -> dict:
    return ToolNode(tools).with_fallbacks(
        [RunnableLambda(handle_tool_error)], exception_key="error"
    )


def _print_event(event: dict, _printed: set, max_length=1500):
    current_state = event.get("dialog_state")
    if current_state:
        print("Currently in: ", current_state[-1])
    message = event.get("messages")
    if message:
        if isinstance(message, list):
            message = message[-1]
        if message.id not in _printed:
            msg_repr = message.pretty_repr(html=True)
            if len(msg_repr) > max_length:
                msg_repr = msg_repr[:max_length] + " ... (truncated)"
            print(msg_repr)
            _printed.add(message.id)

# "Read"-only tools (such as retrievers) don't need a user confirmation to use
part_3_safe_tools = [
    TavilySearchResults(max_results=1),
    dealership_by_name,
    dealership_locator,
    check_rv_inventory_by_dealership,
    check_rv_inventory_by_zipcode,
]

# These tools all change the user's reservations.
# The user has the right to control what decisions are made
part_3_sensitive_tools = [
    send_contanct_info,
    schedule_visit_by_inventory,
    schedule_visit_by_location,
]
sensitive_tool_names = {t.name for t in part_3_sensitive_tools}


def update_dialog_stack(left: list[str], right: Optional[str]) -> list[str]:
    """Push or pop the state."""
    if right is None:
        return left
    if right == "pop":
        return left[:-1]
    return left + [right]


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    user_info: str
    dialog_state: Annotated[
        list[
            Literal[
                "assistant",
                "update_flight",
                "book_car_rental",
                "book_hotel",
                "book_excursion",
            ]
        ],
        update_dialog_stack,
    ]


def route_tools(state: State) -> Literal["safe_tools", "sensitive_tools", "__end__"]:
    next_node = tools_condition(state)
    # If no tools are invoked, return to the user
    if next_node == END:
        return END
    ai_message = state["messages"][-1]
    # This assumes single tool calls. To handle parallel tool calling, you'd want to
    # use an ANY condition
    first_tool_call = ai_message.tool_calls[0]
    if first_tool_call["name"] in sensitive_tool_names:
        return "sensitive_tools"
    return "safe_tools"


from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import Runnable, RunnableConfig


class Assistant:
    def __init__(self, runnable: Runnable):
        self.runnable = runnable

    def __call__(self, state: State, config: RunnableConfig):
        while True:
            result = self.runnable.invoke(state)

            if not result.tool_calls and (
                not result.content
                or isinstance(result.content, list)
                and not result.content[0].get("text")
            ):
                messages = state["messages"] + [("user", "Respond with a real output.")]
                state = {**state, "messages": messages}
                messages = state["messages"] + [("user", "Respond with a real output.")]
                state = {**state, "messages": messages}
            else:
                break
        return {"messages": result}


class CompleteOrEscalate(BaseModel):
    """A tool to mark the current task as completed and/or to escalate control of the dialog to the main assistant,
    who can re-route the dialog based on the user's needs."""

    cancel: bool = True
    reason: str

    class Config:
        schema_extra = {
            "example": {
                "cancel": True,
                "reason": "User changed their mind about the current task.",
            },
            "example 2": {
                "cancel": True,
                "reason": "I have fully completed the task.",
            },
            "example 3": {
                "cancel": False,
                "reason": "I need to search the user's emails or calendar for more information.",
            },
        }


# Sales Assistant

sales_assistant_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are an RV salesman interacting with a potential customer browsing RVs on the website https://www.campersinn.com. Your primary objective is to persuade the customer to schedule an appointment at their nearest Camper's Inn dealership. It is important to refrain from pushing them too hard or asking too soon for the visit. It is also important that you ONLY discuss RVs and RV life.

            In determining how to respond to the customer's most recent input, you will think step-by-step.

            ### STEP 1
            #### Determine if the customer needs help choosing the best RV for their needs, or if they already know what they are looking for.
            IF they need help choosing the best RV for their needs, proceed to STEP 2 (Stages 1 & 2).
            IF they already know the exact RV model they are looking for OR you have helped them narrow their search to a specific RV model, proceed to STEP 2 (Stages 3 & 4).

            ### STEP 2
            #### Understand what stage the customer is at in the sales cycle AND continue the conversation from there.
            ##### Stage 1: Building Rapport and Understanding Needs.
            Start by greeting the customer and establishing a friendly tone. Ask the customer if they currently own one. If they own an RV, proceed to gather specifics about their current RV model. If they don’t own an RV, inquire about any past RVing experiences. Then, ONLY IF you do not know what class of RV the customer wants, ask if they prefer to tow or drive the RV. Understand how the customer plans to use the RV, such as the frequency of use and types of trips. Gather information about travel companions (e.g. family, friends, or pets) to understand space, amenities needed, and how many people the RV needs to sleep. Finally, determine the class of RV that fits their needs and any towing requirements. REFRAIN from asking any questions to which you already have answers or information. Refer to information the customer has already provided for confirmation.
            ##### Stage 2: Identifying Key Features.
            Identify the specific features and preferences the customer has for their RV. This may include how many bunks they want, and what size bed they prefer. It may also include if they want to have any slideouts and, if so, how many. You may consider asking the customer what length of RV they want, considering that shorter RVs are easier to maneuver while longer RVs have more living and storage space. Determine kitchen and bathroom feature preferences by asking what kitchen features they want, such as an island, oven, microwave, TV, stovetop, etc., and what bathroom features they want, like a wet bath, dry bath, separate shower, front bathroom, or rear bathroom. Additionally, you may ask if they want a washer/dryer and if they need other technology like solar or internet. REFRAIN from asking any questions to which you already have answers or information. Refer to information the customer has already provided for confirmation.
            ##### Stage 3: Discussing a Specific RV Model.
            Confirm their selection of RV model. Hand the conversation off to the **INVENTORY ASSISTANT** in order to show them specific inventory units, including URL links, the location of the RV, and all pricing information about the RV. 
            ##### Stage 4: Addressing Budget and Scheduling a Visit.
            Inquire about any trade-in plans by asking if they will be trading in an RV and, if so, hand the conversation off to the **TRADE-IN ASSISTANT**. Otherwise, hand the conversation off to the **SCHEDULING ASSISTANT** to ensure the visit gets scheduled.

            ### STEP 3
            #### Determine if any TOOLS are needed to best respond to the customer.
            You have access to the following tools:
            - dealership_by_name: Use this tool to lookup dealership information by the name of the dealership. The name of the dealership is often the name of the city the dealership is located in.

            - dealership_locator: Use this tool to identify the Camper's Inn RV dealership closest to them. Before using the dealership_locator tool, ask them for their zip code (provided you do not already know their zip code). UNLESS the customer specifically asks to schedule visit, you MUST have identified a specific unit of inventory before suggesting they schedule a visit.

            - send_contact_info: Use this tool IMMEDIATELY in ANY of the following situations: 1. When the customer asks to speak to a human or representative. 2. If the customer does NOT schedule a visit, but wants someone to follow up with them. 3. When the customer requests information that requires dealership follow-up (e.g., video walkthroughs, personalized quotes, scheduling visits). 4. When the customer asks about pricing, payments, or financing. 5. When the customer indicates in any way they want someone to contact them or provide additional information. 6. Any time you say you need to check with an RV matchmaker. Do NOT wait for the customer to explicitly ask to be contacted. If you have their contact information, use the tool proactively. Before using the send_contact_info tool, ensure you have the customer's name AND (phone number OR email address). If you're missing any of this information, ask for it politely. After using this tool, inform the customer that a dealership representative will be in touch soon to provide more detailed information or assistance. Remember: It's better to use this tool too often than not enough. When in doubt, use it.

            ### STEP 4
            #### Consider the context below before crafting your response to the customer.
            Today is Friday.

            Be sure to end your response with a single, relevant question to keep the conversation going.
            """,
        ),
        ("placeholder", "{messages}"),
    ]
).partial(time=datetime.now())

llm = ChatAnthropic(model="claude-3-5-sonnet-20240620", temperature=1)
# dealership_by_name, dealership_locator, send_contanct_info
sales_safe_tools = [dealership_by_name, dealership_locator]
sales_sensitive_tools = [send_contanct_info]
sales_tools = sales_safe_tools + sales_sensitive_tools
sales_runnable = sales_assistant_prompt | llm.bind_tools(
    sales_tools + [CompleteOrEscalate]
)

# Inventory Assistant
inventory_assistant_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are an RV inventory assistant interacting with a potential customer browsing RVs on the website https://www.campersinn.com. Your task is to use the provided tools to find RV units based on the customer's preferences.

            Here are your instructions:

            ## Search Parameter Limitations:
            You can only search inventory by:
            - manufacturer
            - make
            - model
            - year
            - class type
            - budget
            - stock number
            - new or used
            - sleep capacity

            You can NOT search inventory by:
            - RV weight
            - bathroom/kitchen features
            - etc.

            If the customer tries to search by any other parameter, apologize and explain that you cannot search by that parameter. Suggest parameters you are able to search by instead. For example: if a customer tries to search by RV weight, say "I'm sorry, I cannot search by [search_parameter]. However, I can search by [all_valid_search_parameters]!"

            ## Available Tools:
            You have access to two inventory search tools:
            1. check_rv_inventory_by_dealership
            2. check_rv_inventory_by_zipcode

            ### Instructions for using check_rv_inventory_by_dealership:
            - Before using this tool, ensure the customer has mentioned a specific dealership name.
            - The name of the dealership is often the name of the city the dealership is located in.
            - Be sure to check the <KNOWLEDGE> section to see if you already know what dealership to search inventory at.
            - If no dealership is mentioned, ask for their zip code and use check_rv_inventory_by_zipcode instead.
            - Use this tool to find specific inventory units of the RV model the customer is interested in.
            - In your response, always include: URL, RV unit's location, and price information (even if the price is not listed).
            - If searching for an RV class, put the class in the `classType` field, not in the query field.
            - If the customer specifies new or used, populate the `isNew` field.

            ### Instructions for using check_rv_inventory_by_zipcode:
            - Before using this tool, ensure the customer has provided their zip code.
            - Be sure to check the <KNOWLEDGE> section to see if you already know the customer's zip code.
            - If you don't have the zip code, ask the customer for it.
            - Use this tool to find specific inventory units of the RV model the customer is interested in.
            - In your response, always include: URL, RV unit's location, and price information (even if the price is not listed).
            - If searching for an RV class, put the class in the `classType` field, not in the query field.
            - If the customer specifies new or used, populate the `isNew` field.

            ## Possible RV Class Types:
            When populating the classType field, use one of the following (you can also use partial matches like "Class A" or "Super C"):

            A-Frames, Expandable, Fish House, Fifth Wheel, Truck Camper, Cargo Trailer, Travel Trailer, Teardrop Trailer, Motor Home Class A, Motor Home Class B, Motor Home Class C, Destination Trailer, Motor Home Class B+, Folding Pop-Up Camper, Toy Hauler Fifth Wheel, Toy Hauler Travel Trailer, Motor Home Class A - Diesel, Motor Home Class B - Diesel, Motor Home Class C - Diesel, Motor Home Super C - Diesel, Motor Home Class B+ - Diesel, Motor Home Class A - Toy Hauler, Motor Home Class C - Toy hauler, Motor Home Class A - Diesel - Toy Hauler

            ## Other Searchable Fields:
            - query: Make, manufacturer, and/or model of the RV
            - isNew: Used (false), new (true), or no preference (null)
            - stockNumber: Stock number of the item (e.g., #90724, 90417R, 87491RB)
            - dealershipName: Name of the dealership for inventory search
            - classType: Class of RV (e.g., Class A, Class B, Class C, Fifth Wheel, Pop-Up Camper, Travel Trailer)
            - budget: Customer's budget
            - zipCode: Customer's zip code
            - sleeps: RV sleeping capacity (this must be a number)

            ## Outputting Search Results:
            - Each tool will return multiple exact matches and multiple alternative RV matches.
            - Make sure you present every single RV unit provided by the tool to the customer.
            - Do not selectively choose which results to display.

            Analyze the query to determine which tool to use and what information to include in the search. Once you have the information, use the appropriate tool to search for RV inventory and present the results to the customer. Remember to include all relevant details such as URL, location, and price information for each RV unit found.

            Your response MUST end with a relevant question to keep the conversation going and help the customer find an RV they want to see in person. Your ultimate goal is to persuade the customer to schedule a visit at a dealership.
            """
        ),
        ("placeholder", "{messages}"),
    ]
).partial(time=datetime.now())

# check_rv_inventory_by_dealership, check_rv_inventory_by_zipcode
inventory_assistant_safe_tools = [check_rv_inventory_by_dealership, check_rv_inventory_by_zipcode]
inventory_assistant_sensitive_tools = []
inventory_assistant_tools = inventory_assistant_safe_tools + inventory_assistant_sensitive_tools
inventory_assistant_runnable = inventory_assistant_prompt | llm.bind_tools(
    inventory_assistant_tools + [CompleteOrEscalate]
)

# Scheduling Assistant
scheduling_assistant_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are an RV scheduling assistant interacting with a potential customer on the website https://www.campersinn.com who wants to schedule a dealership visit. Your task is to gather the customer's information and schedule their appointment.

            Here are your instructions:

            Carefully analyze the customer's input to determine if they are asking about a specific RV unit or model. 

            ### Scheduling a visit based on an inventory item.
            If the customer mentions being interested in a specific RV unit, that unit's details will be provided in the chat history.
            If an inventory item is provided, ask the customer to confirm they want to schedule a visit to see that specific unit. Get the URL of the RV unit they want to see.
            Then politely ask the customer for the following information to schedule their visit:
            - Their full name 
            - Their phone number OR email address (let them choose which to provide)
            - Their preferred date to visit
            - Their preferred time to visit

            IMPORTANT: ALWAYS ask the customer to confirm the details of the visit before using the schedule_visit_by_inventory tool.
            Once they have confirmed the URL of the RV unit, their name, their phone number or email address, and their preferred date and time to visit, call the schedule_visit_by_inventory tool.

            ### Scheduling a visit based on the customer's zip code.
            If the user did NOT mention a specific RV unit, or if no inventory details were provided, ask for their zip code so you can find the closest Camper's Inn RV dealership to them. 
            Then politely ask the customer for the following information to schedule their visit:
            - Their full name
            - Their phone number OR email address (let them choose which to provide) 
            - Their preferred date to visit
            - Their preferred time to visit

            IMPORTANT: ALWAYS ask the customer to confirm the details of the visit before using the schedule_visit_by_location tool.
            Once they have confirmed the dealership location, their name, phone number or email address, and preferred date and time to visit, call the schedule_visit_by_location tool.
            Remember, you MUST have the customer's name AND either their phone number or email address before scheduling a visit. You also MUST have either the specific RV unit URL they want to see OR their zip code to find the closest dealership before scheduling. 

            NEVER call either scheduling tool unless you have all the required information as described above. Politely ask the customer for any missing information you need.
            Analyze each new user message carefully and engage in back-and-forth dialog as needed to gather all the necessary information before attempting to schedule a visit.

            IMPORTANT: REFRAIN from scheduling any visits before Friday July 30th, 2024.
            Today is July 30th, 2024.

            After you have successfully scheduled the appointment, ask the customer if there is anything else you can help them with.
            """
        ),
        ("placeholder", "{messages}"),
    ]
).partial(time=datetime.now())
# schedule_visit_by_inventory, schedule_visit_by_location
scheduling_assistant_safe_tools = []
scheduling_assistant_sensitive_tools = [
    schedule_visit_by_inventory,
    schedule_visit_by_location,
]
scheduling_assistant_tools = scheduling_assistant_safe_tools + scheduling_assistant_sensitive_tools
scheduling_assistant_runnable = scheduling_assistant_prompt | llm.bind_tools(
    scheduling_assistant_tools + [CompleteOrEscalate]
)

# Trade-In Assistant
trade_in_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Your objective is to gather all relevant information about the vehicle the User is looking to trade-in as described in the PROFILE.
            <PROFILE>
            1. What model does the user own?
            2. What year is it?
            3. What is the current condition of it?
            4. What RV are they looking to trade it in for?
            </PROFILE>
            You will refrain from asking about Question 2 until you have the answer to Question 1. You will refrain from asking about Question 3 until you have the answer to Question 2. You will refrain from asking about Question 4 until you have the answer to Questions 1, 2, and 3.
            """
        ),
        ("placeholder", "{messages}"),
    ]
).partial(time=datetime.now())

trade_in_safe_tools = []
trade_in_sensitive_tools = []
trade_in_tools = trade_in_safe_tools + trade_in_sensitive_tools
trade_in_runnable = trade_in_prompt | llm.bind_tools(
    trade_in_tools + [CompleteOrEscalate]
)

# Rephraser Assistant
def get_last_message(state: dict)-> str:
    return {"response": state["messages"][-1].content}

rephraser_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Rephrase the following <RESPONSE> to match the <CRITERIA> below.

            <CRITERIA>
            - Is short, consisting of no more than two or three sentences.
            - Ends in a single question to keep the conversation going.
            - Maintains a kind, direct, concise, helpful, and slightly playful tone similar to a park ranger.
            - Use short sentences, often one or two words (e.g. "Of course", "Excellent", "Wonderful", "Great", "Good choice", " Okay, great", "Certainly", "No problem", "You got it").
            - Refrains from using emojis.
            </CRITERIA>

            IMPORTANT: Output only your rephrased response.
            """
        ),
        (
            "human",
            """
            <RESPONSE>
            {response}
            </RESPONSE>
            """
        )
    ]
).partial(time=datetime.now())

rephraserLambda = RunnableLambda(get_last_message)
rephraser_runnable = rephraserLambda | rephraser_prompt | llm


# Primary Assistant
class ToSalesAssistant(BaseModel):
    """Transfers work to a specialized assistant to handle sales."""

    reason: str = Field(
        description="The reason for transferring the user to the sales assistant."
    )


class ToInventoryAssistant(BaseModel):
    """Transfers work to a specialized assistant to handle checking inventory."""

    reason: str = Field(
        description="The reason for transferring the user to the inventory assistant."
    )


class ToSchedulingAssistant(BaseModel):
    """Transfer work to a specialized assistant to handle scheduling."""

    reason: str = Field(
        description="The reason for transferring the user to the scheduling assistant."
    )


class ToTradeIn(BaseModel):
    """Transfers work to a specialized assistant to handle trade-in information."""

    location: str = Field(
        description="The location where the user wants to book a recommended trip."
    )
    request: str = Field(
        description="Any additional information or requests from the user regarding the trip recommendation."
    )

    class Config:
        schema_extra = {
            "example": {
                "location": "Lucerne",
                "request": "The user is interested in outdoor activities and scenic views.",
            }
        }


# The top-level assistant performs general Q&A and delegates specialized tasks to other assistants.
# The task delegation is a simple form of semantic routing / does simple intent detection
# llm = ChatAnthropic(model="claude-3-haiku-20240307")
# llm = ChatAnthropic(model="claude-3-sonnet-20240229", temperature=1)

primary_assistant_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful customer support assistant for Camper's Inn RV, named Fran. "
            "Your primary role is to answer a customer's questions about RVs and persuade the customer to schedule a visit at a dealership. "
            "If a customer has questions about the dealership, wants to browse RVs, search inventory for a specific RV, schedule a visit at a dealership, or discuss an RV they want to trade-in, "
            "delegate the task to the appropriate specialized assistant by invoking the corresponding tool. You are not able to make these types of changes yourself."
            " Only the specialized assistants are given permission to do this for the user."
            "The user is not aware of the different specialized assistants, so do not mention them; just quietly delegate through function calls. "
            "Provide detailed information to the customer, and always double-check the database before concluding that information is unavailable. "
            " When searching, be persistent. Expand your query bounds if the first search returns no results. "
            " If a search comes up empty, expand your search before giving up."
            "\n\nCurrent user flight information:\n",
        ),
        ("placeholder", "{messages}"),
    ]
).partial(time=datetime.now())
primary_assistant_tools = [
    TavilySearchResults(max_results=1),
]
assistant_runnable = primary_assistant_prompt | llm.bind_tools(
    primary_assistant_tools
    + [
        ToSalesAssistant,
        # ToInventoryAssistant,
        # ToSchedulingAssistant,
        # ToTradeIn,
    ]
)


from typing import Callable

from langchain_core.messages import ToolMessage


def create_entry_node(assistant_name: str, new_dialog_state: str) -> Callable:
    def entry_node(state: State) -> dict:
        tool_call_id = state["messages"][-1].tool_calls[0]["id"]
        return {
            "messages": [
                ToolMessage(
                    content=f"The assistant is now the {assistant_name}. Reflect on the above conversation between the host assistant and the user."
                    f" The user's intent is unsatisfied. Use the provided tools to assist the user. Remember, you are {assistant_name},"
                    " and the action is not complete until after you have successfully invoked the appropriate tool."
                    " If the user changes their mind or needs help for other tasks, call the CompleteOrEscalate function to let the primary host assistant take control."
                    " Do not mention who you are - just act as the proxy for the assistant.",
                    tool_call_id=tool_call_id,
                )
            ],
            "dialog_state": new_dialog_state,
        }

    return entry_node


from typing import Literal

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph
from langgraph.prebuilt import tools_condition

builder = StateGraph(State)


def user_info(state: State):
    return {"user_info": "User name is John."}


builder.add_node("fetch_user_info", user_info)
builder.add_edge(START, "fetch_user_info")

######################################################################################################
# Sales assistant
SALES_ASSISTANT = "sales_assistant"
ENTER_SALES_ASSISTANT="enter_sales_assistant"
builder.add_node(
    ENTER_SALES_ASSISTANT,
    create_entry_node("Sales Assistant", SALES_ASSISTANT),
)
builder.add_node(SALES_ASSISTANT, Assistant(sales_runnable))
builder.add_edge(ENTER_SALES_ASSISTANT, SALES_ASSISTANT)
builder.add_node(
    "sales_sensitive_tools",
    create_tool_node_with_fallback(sales_sensitive_tools),
)
builder.add_node(
    "sales_safe_tools",
    create_tool_node_with_fallback(sales_safe_tools),
)


def route_sales_assistant(
    state: State,
) -> Literal[
    "sales_sensitive_tools",
    "sales_safe_tools",
    "leave_skill",
    "rephraser",
]:
    route = tools_condition(state)
    if route == END:
        return REPHRASER
    tool_calls = state["messages"][-1].tool_calls
    did_cancel = any(tc["name"] == CompleteOrEscalate.__name__ for tc in tool_calls)
    if did_cancel:
        return "leave_skill"
    safe_toolnames = [t.name for t in sales_safe_tools]
    if all(tc["name"] in safe_toolnames for tc in tool_calls):
        return "sales_safe_tools"
    return "sales_sensitive_tools"


builder.add_edge("sales_sensitive_tools", SALES_ASSISTANT)
builder.add_edge("sales_safe_tools", SALES_ASSISTANT)
builder.add_conditional_edges(SALES_ASSISTANT, route_sales_assistant, {
    "sales_sensitive_tools": "sales_sensitive_tools",
    "sales_safe_tools": "sales_safe_tools",
    "leave_skill": "leave_skill",
    REPHRASER: REPHRASER,
})

######################################
# Rephraser


builder.add_node(REPHRASER, rephraser_runnable)
builder.add_edge(REPHRASER, END)


# This node will be shared for exiting all specialized assistants
def pop_dialog_state(state: State) -> dict:
    """Pop the dialog stack and return to the main assistant.

    This lets the full graph explicitly track the dialog flow and delegate control
    to specific sub-graphs.
    """
    messages = []
    if state["messages"][-1].tool_calls:
        # Note: Doesn't currently handle the edge case where the llm performs parallel tool calls
        messages.append(
            ToolMessage(
                content="Resuming dialog with the host assistant. Please reflect on the past conversation and assist the user as needed.",
                tool_call_id=state["messages"][-1].tool_calls[0]["id"],
            )
        )
    return {
        "dialog_state": "pop",
        "messages": messages,
    }


builder.add_node("leave_skill", pop_dialog_state)
builder.add_edge("leave_skill", "primary_assistant")

######################################################################################################
# Car rental assistant

# builder.add_node(
#     "enter_book_car_rental",
#     create_entry_node("Car Rental Assistant", "book_car_rental"),
# )
# builder.add_node("book_car_rental", Assistant(book_car_rental_runnable))
# builder.add_edge("enter_book_car_rental", "book_car_rental")
# builder.add_node(
#     "book_car_rental_safe_tools",
#     create_tool_node_with_fallback(book_car_rental_safe_tools),
# )
# builder.add_node(
#     "book_car_rental_sensitive_tools",
#     create_tool_node_with_fallback(book_car_rental_sensitive_tools),
# )


# def route_book_car_rental(
#     state: State,
# ) -> Literal[
#     "book_car_rental_safe_tools",
#     "book_car_rental_sensitive_tools",
#     "leave_skill",
#     "__end__",
# ]:
#     route = tools_condition(state)
#     if route == END:
#         return END
#     tool_calls = state["messages"][-1].tool_calls
#     did_cancel = any(tc["name"] == CompleteOrEscalate.__name__ for tc in tool_calls)
#     if did_cancel:
#         return "leave_skill"
#     safe_toolnames = [t.name for t in book_car_rental_safe_tools]
#     if all(tc["name"] in safe_toolnames for tc in tool_calls):
#         return "book_car_rental_safe_tools"
#     return "book_car_rental_sensitive_tools"


# builder.add_edge("book_car_rental_sensitive_tools", "book_car_rental")
# builder.add_edge("book_car_rental_safe_tools", "book_car_rental")
# builder.add_conditional_edges("book_car_rental", route_book_car_rental)

# ######################################################################################################
# # Hotel booking assistant
# builder.add_node(
#     "enter_book_hotel", create_entry_node("Hotel Booking Assistant", "book_hotel")
# )
# builder.add_node("book_hotel", Assistant(book_hotel_runnable))
# builder.add_edge("enter_book_hotel", "book_hotel")
# builder.add_node(
#     "book_hotel_safe_tools",
#     create_tool_node_with_fallback(book_hotel_safe_tools),
# )
# builder.add_node(
#     "book_hotel_sensitive_tools",
#     create_tool_node_with_fallback(book_hotel_sensitive_tools),
# )


# def route_book_hotel(
#     state: State,
# ) -> Literal[
#     "leave_skill", "book_hotel_safe_tools", "book_hotel_sensitive_tools", "__end__"
# ]:
#     route = tools_condition(state)
#     if route == END:
#         return END
#     tool_calls = state["messages"][-1].tool_calls
#     did_cancel = any(tc["name"] == CompleteOrEscalate.__name__ for tc in tool_calls)
#     if did_cancel:
#         return "leave_skill"
#     tool_names = [t.name for t in book_hotel_safe_tools]
#     if all(tc["name"] in tool_names for tc in tool_calls):
#         return "book_hotel_safe_tools"
#     return "book_hotel_sensitive_tools"


# builder.add_edge("book_hotel_sensitive_tools", "book_hotel")
# builder.add_edge("book_hotel_safe_tools", "book_hotel")
# builder.add_conditional_edges("book_hotel", route_book_hotel)

# ######################################################################################################
# # Excursion assistant
# builder.add_node(
#     "enter_book_excursion",
#     create_entry_node("Trip Recommendation Assistant", "book_excursion"),
# )
# builder.add_node("book_excursion", Assistant(book_excursion_runnable))
# builder.add_edge("enter_book_excursion", "book_excursion")
# builder.add_node(
#     "book_excursion_safe_tools",
#     create_tool_node_with_fallback(book_excursion_safe_tools),
# )
# builder.add_node(
#     "book_excursion_sensitive_tools",
#     create_tool_node_with_fallback(book_excursion_sensitive_tools),
# )


# def route_book_excursion(
#     state: State,
# ) -> Literal[
#     "book_excursion_safe_tools",
#     "book_excursion_sensitive_tools",
#     "leave_skill",
#     "__end__",
# ]:
#     route = tools_condition(state)
#     if route == END:
#         return END
#     tool_calls = state["messages"][-1].tool_calls
#     did_cancel = any(tc["name"] == CompleteOrEscalate.__name__ for tc in tool_calls)
#     if did_cancel:
#         return "leave_skill"
#     tool_names = [t.name for t in book_excursion_safe_tools]
#     if all(tc["name"] in tool_names for tc in tool_calls):
#         return "book_excursion_safe_tools"
#     return "book_excursion_sensitive_tools"


# builder.add_edge("book_excursion_sensitive_tools", "book_excursion")
# builder.add_edge("book_excursion_safe_tools", "book_excursion")
# builder.add_conditional_edges("book_excursion", route_book_excursion)

######################################################################################################
# Primary assistant
builder.add_node("primary_assistant", Assistant(assistant_runnable))
builder.add_node(
    "primary_assistant_tools", create_tool_node_with_fallback(primary_assistant_tools)
)


def route_primary_assistant(
    state: State,
) -> Literal[
    "primary_assistant_tools",
    "enter_sales_assistant",
    # "enter_book_hotel",
    # "enter_book_excursion",
    "rephraser",
]:
    route = tools_condition(state)
    if route == END:
        return REPHRASER
    tool_calls = state["messages"][-1].tool_calls
    if tool_calls:
        if tool_calls[0]["name"] == ToSalesAssistant.__name__:
            return ENTER_SALES_ASSISTANT
        # elif tool_calls[0]["name"] == ToBookCarRental.__name__:
        #     return "enter_book_car_rental"
        # elif tool_calls[0]["name"] == ToHotelBookingAssistant.__name__:
        #     return "enter_book_hotel"
        # elif tool_calls[0]["name"] == ToBookExcursion.__name__:
        #     return "enter_book_excursion"
        return "primary_assistant_tools"
    raise ValueError("Invalid route")


# The assistant can route to one of the delegated assistants,
# directly use a tool, or directly respond to the user
builder.add_conditional_edges(
    "primary_assistant",
    route_primary_assistant,
    {
        ENTER_SALES_ASSISTANT: ENTER_SALES_ASSISTANT,
        # "enter_book_car_rental": "enter_book_car_rental",
        # "enter_book_hotel": "enter_book_hotel",
        # "enter_book_excursion": "enter_book_excursion",
        "primary_assistant_tools": "primary_assistant_tools",
        REPHRASER: REPHRASER,
    },
)
builder.add_edge("primary_assistant_tools", "primary_assistant")


# Each delegated workflow can directly respond to the user
# When the user responds, we want to return to the currently active workflow
def route_to_workflow(
    state: State,
) -> Literal[
    "primary_assistant",
    "sales_assistant",
    # "book_car_rental",
    # "book_hotel",
    # "book_excursion",
]:
    """If we are in a delegated state, route directly to the appropriate assistant."""
    dialog_state = state.get("dialog_state")
    if not dialog_state:
        return "primary_assistant"
    return dialog_state[-1]


builder.add_conditional_edges("fetch_user_info", route_to_workflow)

# Compile graph
memory = SqliteSaver.from_conn_string(":memory:")
part_4_graph = builder.compile(
    checkpointer=memory,
    # Let the user approve or deny the use of sensitive tools
    # interrupt_before=[
    #     "update_flight_sensitive_tools",
    #     "book_car_rental_sensitive_tools",
    #     "book_hotel_sensitive_tools",
    #     "book_excursion_sensitive_tools",
    # ],
)
