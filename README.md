# Chatbot

A Streamlit-based chatbot with DeepSeek API support.

## Features

- Multi-turn conversation with context
- Chat history persistence (SQLite)
- File upload and text extraction
- Streaming responses

## Setup

Defaulting to user installation because normal site-packages is not writeable
Requirement already satisfied: streamlit>=1.30.0 in /home/linuxea/.local/lib/python3.10/site-packages (from -r requirements.txt (line 1)) (1.54.0)
Requirement already satisfied: openai>=1.0.0 in /home/linuxea/.local/lib/python3.10/site-packages (from -r requirements.txt (line 2)) (1.76.2)
Requirement already satisfied: python-dotenv>=1.0.0 in /home/linuxea/.local/lib/python3.10/site-packages (from -r requirements.txt (line 3)) (1.2.1)
Requirement already satisfied: watchdog<7,>=2.1.5 in /home/linuxea/.local/lib/python3.10/site-packages (from streamlit>=1.30.0->-r requirements.txt (line 1)) (6.0.0)
Requirement already satisfied: numpy<3,>=1.23 in /home/linuxea/.local/lib/python3.10/site-packages (from streamlit>=1.30.0->-r requirements.txt (line 1)) (2.2.5)
Requirement already satisfied: click<9,>=7.0 in /home/linuxea/.local/lib/python3.10/site-packages (from streamlit>=1.30.0->-r requirements.txt (line 1)) (8.1.8)
Requirement already satisfied: gitpython!=3.1.19,<4,>=3.0.7 in /home/linuxea/.local/lib/python3.10/site-packages (from streamlit>=1.30.0->-r requirements.txt (line 1)) (3.1.46)
Requirement already satisfied: pyarrow>=7.0 in /home/linuxea/.local/lib/python3.10/site-packages (from streamlit>=1.30.0->-r requirements.txt (line 1)) (23.0.0)
Requirement already satisfied: pillow<13,>=7.1.0 in /home/linuxea/.local/lib/python3.10/site-packages (from streamlit>=1.30.0->-r requirements.txt (line 1)) (11.2.1)
Requirement already satisfied: requests<3,>=2.27 in /home/linuxea/.local/lib/python3.10/site-packages (from streamlit>=1.30.0->-r requirements.txt (line 1)) (2.32.3)
Requirement already satisfied: typing-extensions<5,>=4.10.0 in /home/linuxea/.local/lib/python3.10/site-packages (from streamlit>=1.30.0->-r requirements.txt (line 1)) (4.13.2)
Requirement already satisfied: packaging>=20 in /home/linuxea/.local/lib/python3.10/site-packages (from streamlit>=1.30.0->-r requirements.txt (line 1)) (25.0)
Requirement already satisfied: blinker<2,>=1.5.0 in /home/linuxea/.local/lib/python3.10/site-packages (from streamlit>=1.30.0->-r requirements.txt (line 1)) (1.9.0)
Requirement already satisfied: protobuf<7,>=3.20 in /home/linuxea/.local/lib/python3.10/site-packages (from streamlit>=1.30.0->-r requirements.txt (line 1)) (6.31.1)
Requirement already satisfied: pandas<3,>=1.4.0 in /home/linuxea/.local/lib/python3.10/site-packages (from streamlit>=1.30.0->-r requirements.txt (line 1)) (2.3.3)
Requirement already satisfied: tornado!=6.5.0,<7,>=6.0.3 in /home/linuxea/.local/lib/python3.10/site-packages (from streamlit>=1.30.0->-r requirements.txt (line 1)) (6.5.4)
Requirement already satisfied: tenacity<10,>=8.1.0 in /home/linuxea/.local/lib/python3.10/site-packages (from streamlit>=1.30.0->-r requirements.txt (line 1)) (9.1.2)
Requirement already satisfied: altair!=5.4.0,!=5.4.1,<7,>=4.0 in /home/linuxea/.local/lib/python3.10/site-packages (from streamlit>=1.30.0->-r requirements.txt (line 1)) (6.0.0)
Requirement already satisfied: pydeck<1,>=0.8.0b4 in /home/linuxea/.local/lib/python3.10/site-packages (from streamlit>=1.30.0->-r requirements.txt (line 1)) (0.9.1)
Requirement already satisfied: toml<2,>=0.10.1 in /home/linuxea/.local/lib/python3.10/site-packages (from streamlit>=1.30.0->-r requirements.txt (line 1)) (0.10.2)
Requirement already satisfied: cachetools<7,>=5.5 in /home/linuxea/.local/lib/python3.10/site-packages (from streamlit>=1.30.0->-r requirements.txt (line 1)) (6.2.6)
Requirement already satisfied: pydantic<3,>=1.9.0 in /home/linuxea/.local/lib/python3.10/site-packages (from openai>=1.0.0->-r requirements.txt (line 2)) (2.11.4)
Requirement already satisfied: anyio<5,>=3.5.0 in /home/linuxea/.local/lib/python3.10/site-packages (from openai>=1.0.0->-r requirements.txt (line 2)) (4.9.0)
Requirement already satisfied: sniffio in /home/linuxea/.local/lib/python3.10/site-packages (from openai>=1.0.0->-r requirements.txt (line 2)) (1.3.1)
Requirement already satisfied: distro<2,>=1.7.0 in /home/linuxea/.local/lib/python3.10/site-packages (from openai>=1.0.0->-r requirements.txt (line 2)) (1.9.0)
Requirement already satisfied: jiter<1,>=0.4.0 in /home/linuxea/.local/lib/python3.10/site-packages (from openai>=1.0.0->-r requirements.txt (line 2)) (0.9.0)
Requirement already satisfied: tqdm>4 in /home/linuxea/.local/lib/python3.10/site-packages (from openai>=1.0.0->-r requirements.txt (line 2)) (4.67.1)
Requirement already satisfied: httpx<1,>=0.23.0 in /home/linuxea/.local/lib/python3.10/site-packages (from openai>=1.0.0->-r requirements.txt (line 2)) (0.28.1)
Requirement already satisfied: narwhals>=1.27.1 in /home/linuxea/.local/lib/python3.10/site-packages (from altair!=5.4.0,!=5.4.1,<7,>=4.0->streamlit>=1.30.0->-r requirements.txt (line 1)) (2.16.0)
Requirement already satisfied: jsonschema>=3.0 in /usr/lib/python3/dist-packages (from altair!=5.4.0,!=5.4.1,<7,>=4.0->streamlit>=1.30.0->-r requirements.txt (line 1)) (3.2.0)
Requirement already satisfied: jinja2 in /home/linuxea/.local/lib/python3.10/site-packages (from altair!=5.4.0,!=5.4.1,<7,>=4.0->streamlit>=1.30.0->-r requirements.txt (line 1)) (3.1.6)
Requirement already satisfied: idna>=2.8 in /home/linuxea/.local/lib/python3.10/site-packages (from anyio<5,>=3.5.0->openai>=1.0.0->-r requirements.txt (line 2)) (3.10)
Requirement already satisfied: exceptiongroup>=1.0.2 in /home/linuxea/.local/lib/python3.10/site-packages (from anyio<5,>=3.5.0->openai>=1.0.0->-r requirements.txt (line 2)) (1.2.2)
Requirement already satisfied: gitdb<5,>=4.0.1 in /home/linuxea/.local/lib/python3.10/site-packages (from gitpython!=3.1.19,<4,>=3.0.7->streamlit>=1.30.0->-r requirements.txt (line 1)) (4.0.12)
Requirement already satisfied: httpcore==1.* in /home/linuxea/.local/lib/python3.10/site-packages (from httpx<1,>=0.23.0->openai>=1.0.0->-r requirements.txt (line 2)) (1.0.9)
Requirement already satisfied: certifi in /home/linuxea/.local/lib/python3.10/site-packages (from httpx<1,>=0.23.0->openai>=1.0.0->-r requirements.txt (line 2)) (2025.4.26)
Requirement already satisfied: h11>=0.16 in /home/linuxea/.local/lib/python3.10/site-packages (from httpcore==1.*->httpx<1,>=0.23.0->openai>=1.0.0->-r requirements.txt (line 2)) (0.16.0)
Requirement already satisfied: python-dateutil>=2.8.2 in /home/linuxea/.local/lib/python3.10/site-packages (from pandas<3,>=1.4.0->streamlit>=1.30.0->-r requirements.txt (line 1)) (2.9.0.post0)
Requirement already satisfied: tzdata>=2022.7 in /home/linuxea/.local/lib/python3.10/site-packages (from pandas<3,>=1.4.0->streamlit>=1.30.0->-r requirements.txt (line 1)) (2025.3)
Requirement already satisfied: pytz>=2020.1 in /usr/lib/python3/dist-packages (from pandas<3,>=1.4.0->streamlit>=1.30.0->-r requirements.txt (line 1)) (2022.1)
Requirement already satisfied: typing-inspection>=0.4.0 in /home/linuxea/.local/lib/python3.10/site-packages (from pydantic<3,>=1.9.0->openai>=1.0.0->-r requirements.txt (line 2)) (0.4.0)
Requirement already satisfied: pydantic-core==2.33.2 in /home/linuxea/.local/lib/python3.10/site-packages (from pydantic<3,>=1.9.0->openai>=1.0.0->-r requirements.txt (line 2)) (2.33.2)
Requirement already satisfied: annotated-types>=0.6.0 in /home/linuxea/.local/lib/python3.10/site-packages (from pydantic<3,>=1.9.0->openai>=1.0.0->-r requirements.txt (line 2)) (0.7.0)
Requirement already satisfied: urllib3<3,>=1.21.1 in /home/linuxea/.local/lib/python3.10/site-packages (from requests<3,>=2.27->streamlit>=1.30.0->-r requirements.txt (line 1)) (2.4.0)
Requirement already satisfied: charset-normalizer<4,>=2 in /home/linuxea/.local/lib/python3.10/site-packages (from requests<3,>=2.27->streamlit>=1.30.0->-r requirements.txt (line 1)) (3.4.2)
Requirement already satisfied: smmap<6,>=3.0.1 in /home/linuxea/.local/lib/python3.10/site-packages (from gitdb<5,>=4.0.1->gitpython!=3.1.19,<4,>=3.0.7->streamlit>=1.30.0->-r requirements.txt (line 1)) (5.0.2)
Requirement already satisfied: MarkupSafe>=2.0 in /home/linuxea/.local/lib/python3.10/site-packages (from jinja2->altair!=5.4.0,!=5.4.1,<7,>=4.0->streamlit>=1.30.0->-r requirements.txt (line 1)) (3.0.2)
Requirement already satisfied: six>=1.5 in /usr/lib/python3/dist-packages (from python-dateutil>=2.8.2->pandas<3,>=1.4.0->streamlit>=1.30.0->-r requirements.txt (line 1)) (1.16.0)

      👋 Welcome to Streamlit!

      If you'd like to receive helpful onboarding emails, news, offers, promotions,
      and the occasional swag, please enter your email address below. Otherwise,
      leave this field blank.

      Email:  

