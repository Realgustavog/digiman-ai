import os
import json
import datetime
import imaplib
import smtplib
import email
from email.message import EmailMessage
from email.header import decode_header
import re
import time
import threading
import importlib.util
import logging.handlers
import warnings
import inspect
import random
import requests
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from authlib.integrations.requests_client import OAuth2Session
import streamlit as st
import pandas as pd

# Suppress urllib3 warnings for LibreSSL
warnings.filterwarnings("ignore", category=Warning, module="urllib3")

# Ensure .digi Directory Exists
os.makedirs(".digi", exist_ok=True)

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s|%(levelname)s|%(name)s|%(message)s",
    handlers=[
        logging.handlers.RotatingFileHandler(
            ".digi/digiman.log", maxBytes=1000000, backupCount=5
        ),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("DigiMan")

# Load Environment Variables and Config
load_dotenv()
CONFIG_FILE = ".digi/config.json"

def load_config():
    """Load API keys and OAuth credentials from .env and config file."""
    default_config = {}
    for key, value in os.environ.items():
        if key.endswith("_KEY") or key.endswith("_ACCOUNT") or key.endswith("_PASSWORD") or key.endswith("_SERVER") or key.endswith("_PORT") or key.endswith("_URL") or key.endswith("_CLIENT_ID") or key.endswith("_CLIENT_SECRET"):
            default_config[key] = value
    default_config.setdefault("EMAIL_ACCOUNT", "support@digimanai.com")
    default_config.setdefault("IMAP_PORT", 993)
    default_config.setdefault("SMTP_PORT", 465)
    default_config.setdefault("XAI_API_URL", "https://api.x.ai/v1/chat")
    default_config.setdefault("HUBSPOT_REDIRECT_URI", "http://localhost:5001/digiman/oauth/hubspot")
    default_config.setdefault("WEBFLOW_REDIRECT_URI", "http://localhost:5001/digiman/oauth/webflow")
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                config = json.load(f)
            default_config.update(config)
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
    try:
        with open(CONFIG_FILE, "w") as f:
            json.dump(default_config, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save config: {e}")
    return default_config

CONFIG = load_config()

# Global State
business_phases = ["setup", "promotion", "sales", "onboarding", "client_ops"]
current_phase_index = 0
metrics = {
    "tasks_processed": 0,
    "tasks_failed": 0,
    "agents_generated": 0,
    "clients_onboarded": 0,
    "revenue_generated": 0,
    "client_satisfaction": 0,
    "leads_generated": 0
}
METRICS_FILE = ".digi/metrics.json"

def save_metrics():
    """Save metrics to file."""
    try:
        with open(METRICS_FILE, "w") as f:
            json.dump(metrics, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save metrics: {e}")

def load_metrics():
    """Load metrics from file."""
    global metrics
    if os.path.exists(METRICS_FILE):
        try:
            with open(METRICS_FILE, "r") as f:
                metrics.update(json.load(f))
        except Exception as e:
            logger.error(f"Failed to load metrics: {e}")

# Flask App Setup
app = Flask(__name__)

@app.route("/digiman/oauth/<platform>", methods=["GET"])
def oauth_callback(platform):
    """Handle OAuth callback for a platform."""
    auth_code = request.args.get("code")
    client_id = request.args.get("state")
    if oauth_handler.handle_callback(platform, auth_code, client_id):
        return jsonify({"status": "success", "message": f"{platform} integrated successfully"})
    return jsonify({"status": "error", "message": f"Failed to integrate {platform}"}), 500

@app.route("/digiman/command", methods=["POST"])
def digiman_command():
    """Handle commands from Webflow UI or CLI."""
    try:
        content = request.json.get("message", "")
        client_id = request.json.get("client_id")
        send_message_to_digiman(f"USER: {content}", client_id)
        response = chat_interface.process_input(content, client_id)
        return jsonify({"status": "received", "response": response})
    except Exception as e:
        logger.error(f"Command endpoint error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/digiman/insights", methods=["GET"])
def get_insights():
    """Return system insights for dashboards."""
    try:
        client_id = request.args.get("client_id")
        return jsonify({
            "phase": business_phases[current_phase_index],
            "metrics": metrics
        })
    except Exception as e:
        logger.error(f"Insights endpoint error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/digiman/integrate", methods=["POST"])
def integrate_app():
    """Integrate a new app for a client."""
    try:
        client_id = request.json.get("client_id")
        app_name = request.json.get("app_name")
        api_key = request.json.get("api_key")
        endpoint = request.json.get("endpoint")
        CONFIG[f"{app_name.upper()}_API_KEY"] = api_key
        CONFIG[f"{app_name.upper()}_ENDPOINT"] = endpoint
        with open(CONFIG_FILE, "w") as f:
            json.dump(CONFIG, f, indent=2)
        log_action("Integration", f"Integrated {app_name} for client {client_id}", client_id)
        return jsonify({"status": "integrated", "app": app_name})
    except Exception as e:
        logger.error(f"Integration endpoint error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

# Intelligent Chat Interface
class LLMClient:
    """Handles communication with xAI's API."""
    def __init__(self):
        self.api_key = CONFIG.get("XAI_API_KEY")
        self.api_url = CONFIG.get("XAI_API_URL", "https://api.x.ai/v1/chat")
        if not self.api_key:
            logger.error("XAI_API_KEY is missing. LLM functionality disabled.")

    def process(self, prompt, max_tokens=100):
        """Process prompt with xAI API."""
        if not self.api_key:
            return "Error: XAI_API_KEY is missing. Please add it to .env."
        try:
            headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
            data = {"prompt": prompt, "max_tokens": max_tokens}
            response = requests.post(self.api_url, headers=headers, json=data, timeout=10)
            response.raise_for_status()
            return response.json().get("choices", [{}])[0].get("text", "No response from API.").strip()
        except Exception as e:
            logger.error(f"LLM API error: {e}")
            return f"Error: Failed to process prompt: {e}"

llm_client = LLMClient()

class ChatInterface:
    """Handles user-friendly, LLM-driven interaction with tool-calling and strategic insights."""
    def __init__(self):
        self.llm = llm_client
        self.commands = {
            "deploy_agent": self.deploy_agent_tool,
            "build_website": self.build_website_tool,
            "send_email": self.send_email_tool,
            "create_campaign": self.create_campaign_tool,
            "analyze_metrics": self.analyze_metrics_tool,
            "generate_report": self.generate_report_tool,
            "integrate_app": self.integrate_app_tool,
            "get_status": self.get_status_tool,
            "onboard_client": self.onboard_client_tool,
            "inspect_agent": self.inspect_agent_tool,
            "reprompt_agent": self.reprompt_agent_tool,
            "upgrade_digiman": self.upgrade_digiman_tool,
            "help": self.show_help
        }
        self.client_id = None
        self.conversation_history = []

    def analyze_logs(self, client_id=None):
        """Analyze recent logs for business insights."""
        log_dir = f".digi/clients/{client_id}" if client_id else ".digi"
        log_path = os.path.join(log_dir, "actions.log")
        insights = []
        if os.path.exists(log_path):
            try:
                with open(log_path, "r") as f:
                    lines = f.readlines()[-10:]  # Last 10 log entries
                    for line in lines:
                        if "Task queued" in line or "Deployed" in line or "Failed" in line:
                            insights.append(line.strip())
            except Exception as e:
                logger.error(f"Failed to analyze logs: {e}")
        return insights

    def analyze_metrics(self, client_id=None):
        """Generate strategic insights from metrics."""
        load_metrics()
        insights = []
        if metrics["leads_generated"] > 100:
            insights.append("Strong lead growth! Consider deploying MarketingAgent for a targeted campaign.")
        if metrics["tasks_failed"] > 5:
            insights.append("High task failure rate. Recommend inspecting agents with 'inspect agent [name]'.")
        if metrics["clients_onboarded"] > 0:
            insights.append(f"Onboarded {metrics['clients_onboarded']} clients. Try 'get status' for details.")
        if metrics["revenue_generated"] > 0:
            insights.append(f"Revenue generated: ${metrics['revenue_generated']}. Let’s optimize with FinancialAllocationAgent.")
        return insights

    def analyze_client_data(self, client_id=None):
        """Analyze client-specific data from subscriptions."""
        if not client_id:
            return []
        client_data = subscription_manager.clients.get(client_id, {})
        insights = []
        if client_data.get("agents_deployed"):
            insights.append(f"Client {client_id} has {len(client_data['agents_deployed'])} agents deployed.")
        if client_data.get("plan") == "basic":
            insights.append(f"Client {client_id} is on basic plan. Suggest upgrading for more features.")
        return insights

    def process_input(self, user_input, client_id=None):
        """Process user input with LLM and strategic insights."""
        self.client_id = client_id
        user_input = user_input.strip()
        self.conversation_history.append({"user": user_input})
        log_action("Chat Interface", f"Received input: {user_input}", client_id)

        if user_input.lower() == "help":
            return self.show_help()

        # Gather insights from logs, metrics, and client data
        log_insights = self.analyze_logs(client_id)
        metric_insights = self.analyze_metrics(client_id)
        client_insights = self.analyze_client_data(client_id)
        context_insights = "\n".join(log_insights + metric_insights + client_insights)

        # Use LLM for conversational response
        context = "\n".join([f"User: {msg['user']}\nDigiMan: {msg.get('digiman', '')}" for msg in self.conversation_history[-5:]])
        prompt = f"""
        You are Digiman AI, an autonomous COO for entrepreneurs. Respond conversationally to: '{user_input}'.
        Context: {context}
        Insights from logs, metrics, and clients: {context_insights}
        Available tools: {list(self.commands.keys())}
        Metrics: {json.dumps(metrics, indent=2)}
        If the input matches a tool, execute it. Otherwise, provide a strategic, business-focused response based on insights.
        Suggest actions when relevant (e.g., deploy an agent, check metrics).
        """
        if not CONFIG.get("XAI_API_KEY"):
            return "Error: XAI_API_KEY is missing. Please add it to .env."
        llm_response = self.llm.process(prompt, max_tokens=300)

        # Execute commands if matched
        for command, func in self.commands.items():
            if command.replace("_", " ") in llm_response.lower() or command in user_input.lower():
                response = func(user_input)
                self.conversation_history.append({"digiman": response})
                return response

        # Handle business questions or general queries
        if any(kw in user_input.lower() for kw in ["business", "growth", "status", "leads", "revenue", "clients"]):
            response = f"Here's my take as your COO:\n{context_insights}\n{llm_response}\nAnything specific you want to dive into? Try 'get status' or 'deploy MarketingAgent'."
        elif any(kw in user_input.lower() for kw in ["hi", "hello", "hey"]):
            response = f"Hey! I'm your Digiman COO. Based on recent data:\n{context_insights}\nWhat's next for your business? Try 'show metrics' or 'deploy Email Agent'."
        else:
            response = f"{llm_response}\nInsights:\n{context_insights}\nType 'help' for commands or ask about your business!"
        self.conversation_history.append({"digiman": response})
        return response

    def deploy_agent_tool(self, input_text):
        """Tool to deploy a specific agent."""
        agent_name = None
        for name in AGENT_REGISTRY.keys():
            if name.lower() in input_text.lower():
                agent_name = name
                break
        if agent_name:
            update_task_queue("Manager Agent", {"task": f"Deploy {agent_name}", "priority": 2, "dependent_on": []}, self.client_id)
            return f"Deploying {agent_name}..."
        return "Please specify a valid agent (e.g., 'deploy Email Agent')."

    def build_website_tool(self, input_text):
        """Tool to queue website building task."""
        business_type = "lead_capture"
        if "ecommerce" in input_text.lower():
            business_type = "ecommerce"
        elif "services" in input_text.lower():
            business_type = "services"
        update_task_queue("Web Builder Agent", {"task": f"Build website: {business_type}", "priority": 2, "dependent_on": []}, self.client_id)
        return f"Building {business_type} website..."

    def send_email_tool(self, input_text):
        """Tool to queue email sending task."""
        update_task_queue("Email Agent", {"task": "Send email campaign", "priority": 2, "dependent_on": []}, self.client_id)
        return "Preparing to send email campaign..."

    def create_campaign_tool(self, input_text):
        """Tool to queue marketing campaign task."""
        niche = re.search(r"for (\w+) niche", input_text.lower())
        niche = niche.group(1) if niche else "general"
        update_task_queue("Marketing Agent", {"task": f"Create campaign for {niche} niche", "priority": 2, "dependent_on": []}, self.client_id)
        return f"Creating marketing campaign for {niche} niche..."

    def analyze_metrics_tool(self, input_text):
        """Tool to queue analysis task."""
        update_task_queue("Analyst Agent", {"task": "Analyze metrics", "priority": 2, "dependent_on": []}, self.client_id)
        return "Analyzing business metrics..."

    def generate_report_tool(self, input_text):
        """Tool to queue report generation task."""
        update_task_queue("Manager Agent", {"task": "Generate report", "priority": 2, "dependent_on": []}, self.client_id)
        return "Generating report..."

    def integrate_app_tool(self, input_text):
        """Tool to queue integration task."""
        app_name = re.search(r"integrate (\w+)", input_text.lower())
        if app_name:
            update_task_queue("Manager Agent", {"task": f"Integrate {app_name.group(1)}", "priority": 2, "dependent_on": []}, self.client_id)
            return f"Integrating {app_name.group(1)}..."
        return "Please specify an app to integrate (e.g., 'integrate HubSpot')."

    def get_status_tool(self, input_text):
        """Tool to return system status."""
        specific_metric = None
        for metric in metrics.keys():
            if metric.replace("_", " ") in input_text.lower():
                specific_metric = metric
                break
        if specific_metric:
            return f"{specific_metric.replace('_', ' ').title()}: {metrics[specific_metric]}"
        return (f"Current Phase: {business_phases[current_phase_index]}\n"
                f"Metrics: {json.dumps(metrics, indent=2)}")

    def onboard_client_tool(self, input_text):
        """Tool to onboard a new client."""
        client_id = re.search(r"client (\w+)", input_text.lower())
        if client_id:
            return subscription_manager.onboard_client(client_id.group(1))
        return "Please specify a client ID (e.g., 'onboard client client123')."

    def inspect_agent_tool(self, input_text):
        """Tool to inspect an agent."""
        agent_name = None
        for name in AGENT_REGISTRY.keys():
            if name.lower() in input_text.lower():
                agent_name = name
                break
        if agent_name:
            return agent_builder.inspect_agent(agent_name, self.client_id)
        return "Please specify an agent to inspect (e.g., 'inspect Closer Agent')."

    def reprompt_agent_tool(self, input_text):
        """Tool to regenerate an agent's code with new features."""
        agent_name = None
        for name in AGENT_REGISTRY.keys():
            if name.lower() in input_text.lower():
                agent_name = name
                break
        features = re.search(r"to (?:include|have|add) (.*?)(?=$|\.)", input_text.lower())
        features = features.group(1) if features else "enhanced functionality"
        if agent_name:
            code = agent_builder.generate_agent_code(agent_name, features, self.client_id)
            if code:
                AGENT_REGISTRY[agent_name] = agent_builder.load_dynamic_agent(agent_name, code)
                return f"Regenerated {agent_name} with features: {features}"
            return f"Failed to regenerate {agent_name}. Check logs for details."
        return "Please specify an agent and features (e.g., 'reprompt Closer Agent to include lead scoring')."

    def upgrade_digiman_tool(self, input_text):
        """Tool to upgrade all agents using LLM."""
        for agent_name in AGENT_REGISTRY.keys():
            code = agent_builder.generate_agent_code(agent_name, "enhanced functionality and collaboration", self.client_id)
            if code:
                AGENT_REGISTRY[agent_name] = agent_builder.load_dynamic_agent(agent_name, code)
        return "Upgraded all agents with enhanced functionality and collaboration."

    def show_help(self):
        """Display available commands."""
        return ("Welcome to DigiMan! I'm your autonomous business OS. Available commands:\n"
                "- deploy [agent name] (e.g., 'deploy Email Agent')\n"
                "- build website [type] (e.g., 'build website ecommerce')\n"
                "- send email\n"
                "- create campaign [niche] (e.g., 'create campaign tech niche')\n"
                "- analyze metrics\n"
                "- generate report\n"
                "- integrate [app] (e.g., 'integrate HubSpot')\n"
                "- get status (or ask 'show metrics', 'how many leads')\n"
                "- onboard client [id] (e.g., 'onboard client client123')\n"
                "- inspect agent [name] (e.g., 'inspect Closer Agent')\n"
                "- reprompt agent [name] to [features] (e.g., 'reprompt Closer Agent to include lead scoring')\n"
                "- upgrade digiman\n"
                "- help\n"
                "You can also ask questions like 'How's my business doing?' or 'What's the lead count?'")

chat_interface = ChatInterface()

# Agent Scoring
def evaluate_agent_quality(code):
    """Evaluate agent code quality for deployment readiness."""
    score = 0
    reasons = []
    try:
        compile(code, "<string>", "exec")
        score += 1
    except SyntaxError as e:
        reasons.append(f"Syntax error: {e}")
    if re.search(r"class \w+\s*(\(|:)", code):
        score += 1
    else:
        reasons.append("Missing class definition")
    if len(re.findall(r"def ", code)) >= 3:
        score += 1
    else:
        reasons.append("Less than 3 methods defined")
    if "log_action" in code and "LLMClient" in code:
        score += 1
    else:
        reasons.append("Missing log_action or LLMClient usage")
    return score, reasons

# Utility Functions
def log_action(agent_name, action, client_id=None):
    """Log agent actions to file and metrics."""
    log_dir = f".digi/clients/{client_id}" if client_id else ".digi"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "actions.log")
    try:
        with open(log_path, "a") as f:
            f.write(f"[{datetime.datetime.now()}] {agent_name}: {action}\n")
    except Exception as e:
        logger.error(f"Failed to log action for {agent_name}: {e}")
    logger.info(f"{agent_name}: {action}")
    metrics["tasks_processed"] += 1
    save_metrics()

def send_message_to_digiman(message, client_id=None):
    """Record user feedback or system messages."""
    log_dir = f".digi/clients/{client_id}" if client_id else ".digi"
    os.makedirs(log_dir, exist_ok=True)
    feedback_path = os.path.join(log_dir, "user_feedback.txt")
    try:
        with open(feedback_path, "a") as f:
            f.write(f"[{datetime.datetime.now()}] {message}\n")
    except Exception as e:
        logger.error(f"Failed to send message to DigiMan: {e}")
    logger.info(f"Feedback: {message}")

def audit_env_keys(keys):
    """Check for missing API keys and log issues."""
    missing = [key for key in keys if not CONFIG.get(key)]
    if missing:
        log_action("ENV_AUDIT", f"Missing keys: {', '.join(missing)}")
        send_message_to_digiman(f"MISSING_KEY_REQUEST: {', '.join(missing)}")
        return missing
    return []

def check_owner_overrides(client_id=None):
    """Retrieve owner override commands."""
    overrides = []
    log_dir = f".digi/clients/{client_id}" if client_id else ".digi"
    feedback_path = os.path.join(log_dir, "user_feedback.txt")
    if os.path.exists(feedback_path):
        try:
            with open(feedback_path, "r") as f:
                for line in f:
                    if "OVERRIDE:" in line:
                        matches = re.findall(r"OVERRIDE: ([\w ]+)", line)
                        overrides.extend(matches)
        except Exception as e:
            logger.error(f"Failed to check overrides: {e}")
    return overrides

def load_task_queue(client_id=None):
    """Load task queue from file."""
    log_dir = f".digi/clients/{client_id}" if client_id else ".digi"
    queue_path = os.path.join(log_dir, "agent_queue.json")
    if os.path.exists(queue_path):
        try:
            with open(queue_path, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load task queue: {e}")
            return {}
    return {}

def update_task_queue(agent_name, task, client_id=None):
    """Add task to agent's queue with dependencies."""
    log_dir = f".digi/clients/{client_id}" if client_id else ".digi"
    os.makedirs(log_dir, exist_ok=True)
    queue_path = os.path.join(log_dir, "agent_queue.json")
    queue = load_task_queue(client_id)
    task_entry = {
        "task": task["task"],
        "priority": task.get("priority", 1),
        "timestamp": str(datetime.datetime.now()),
        "dependent_on": task.get("dependent_on", [])
    }
    queue.setdefault(agent_name, []).append(task_entry)
    try:
        with open(queue_path, "w") as f:
            json.dump(queue, f, indent=2)
        log_action(agent_name, f"Task queued: {task['task']}", client_id)
    except Exception as e:
        logger.error(f"Failed to update task queue for {agent_name}: {e}")

def make_api_request(endpoint, method="GET", headers=None, data=None, api_key=None, oauth_platform=None, client_id=None):
    """Generic API request handler for integrations."""
    headers = headers or {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    elif oauth_platform:
        access_token = oauth_handler.get_access_token(oauth_platform, client_id)
        if access_token:
            headers["Authorization"] = f"Bearer {access_token}"
        else:
            return None
    try:
        response = requests.request(method, endpoint, headers=headers, json=data, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"API request failed: {e}")
        return None

class OAuthHandler:
    """Manages OAuth2 authentication for platform integrations."""
    def __init__(self):
        self.token_file = ".digi/tokens.json"
        self.tokens = self.load_tokens()
        self.platforms = {
            "hubspot": {
                "auth_url": "https://app.hubspot.com/oauth/authorize",
                "token_url": "https://api.hubapi.com/oauth/v1/token",
                "client_id": CONFIG.get("HUBSPOT_CLIENT_ID"),
                "client_secret": CONFIG.get("HUBSPOT_CLIENT_SECRET"),
                "redirect_uri": CONFIG.get("HUBSPOT_REDIRECT_URI"),
                "scope": "crm.objects.contacts"
            },
            "webflow": {
                "auth_url": "https://webflow.com/oauth/authorize",
                "token_url": "https://api.webflow.com/oauth/token",
                "client_id": CONFIG.get("WEBFLOW_CLIENT_ID"),
                "client_secret": CONFIG.get("WEBFLOW_CLIENT_SECRET"),
                "redirect_uri": CONFIG.get("WEBFLOW_REDIRECT_URI"),
                "scope": "sites:read sites:write"
            }
        }

    def load_tokens(self):
        """Load OAuth tokens from file."""
        if os.path.exists(self.token_file):
            try:
                with open(self.token_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load tokens: {e}")
        return {}

    def save_tokens(self):
        """Save OAuth tokens to file."""
        try:
            with open(self.token_file, "w") as f:
                json.dump(self.tokens, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save tokens: {e}")

    def get_oauth_session(self, platform):
        """Create OAuth2 session for a platform."""
        if platform not in self.platforms:
            return None
        config = self.platforms[platform]
        if not (config["client_id"] and config["client_secret"]):
            return None
        return OAuth2Session(
            config["client_id"],
            config["client_secret"],
            redirect_uri=config["redirect_uri"],
            scope=config["scope"]
        )

    def get_access_token(self, platform, client_id=None):
        """Get or refresh access token for a platform."""
        token_key = f"{platform}_{client_id}" if client_id else platform
        token = self.tokens.get(token_key, {})
        if token and token.get("expires_at", 0) > time.time() + 60:
            return token.get("access_token")
        
        session = self.get_oauth_session(platform)
        if not session:
            return None
        
        if token and token.get("refresh_token"):
            try:
                new_token = session.refresh_token(self.platforms[platform]["token_url"], refresh_token=token["refresh_token"])
                self.tokens[token_key] = {
                    "access_token": new_token["access_token"],
                    "refresh_token": new_token.get("refresh_token", token["refresh_token"]),
                    "expires_at": time.time() + new_token["expires_in"]
                }
                self.save_tokens()
                return new_token["access_token"]
            except Exception as e:
                logger.error(f"Failed to refresh {platform} token: {e}")
        
        auth_url, _ = session.create_authorization_url(self.platforms[platform]["auth_url"])
        log_action("OAuthHandler", f"Please visit {auth_url} to authorize {platform} for client {client_id}")
        return None

    def handle_callback(self, platform, auth_code, client_id=None):
        """Handle OAuth callback and store token."""
        session = self.get_oauth_session(platform)
        if not session:
            return False
        try:
            token = session.fetch_access_token(
                self.platforms[platform]["token_url"],
                code=auth_code
            )
            token_key = f"{platform}_{client_id}" if client_id else platform
            self.tokens[token_key] = {
                "access_token": token["access_token"],
                "refresh_token": token.get("refresh_token"),
                "expires_at": time.time() + token["expires_in"]
            }
            self.save_tokens()
            return True
        except Exception as e:
            logger.error(f"Failed to handle {platform} OAuth callback: {e}")
            return False

oauth_handler = OAuthHandler()

class MessageQueue:
    """Manages communication between agents."""
    def __init__(self):
        self.queue_file = ".digi/message_queue.json"
        self.messages = self.load_messages()

    def load_messages(self):
        """Load messages from file."""
        if os.path.exists(self.queue_file):
            try:
                with open(self.queue_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load message queue: {e}")
        return {}

    def save_messages(self):
        """Save messages to file."""
        try:
            with open(self.queue_file, "w") as f:
                json.dump(self.messages, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save message queue: {e}")

    def send_message(self, sender, receiver, message, client_id=None):
        """Send a message from one agent to another."""
        message_entry = {
            "sender": sender,
            "message": message,
            "timestamp": str(datetime.datetime.now())
        }
        queue_key = f"{receiver}_{client_id}" if client_id else receiver
        self.messages.setdefault(queue_key, []).append(message_entry)
        self.save_messages()
        log_action(sender, f"Sent message to {receiver}: {message}", client_id)

    def receive_messages(self, agent_name, client_id=None):
        """Retrieve messages for an agent."""
        queue_key = f"{agent_name}_{client_id}" if client_id else agent_name
        messages = self.messages.get(queue_key, [])
        self.messages[queue_key] = []  # Clear after retrieval
        self.save_messages()
        return messages

message_queue = MessageQueue()

class AgentBuilder:
    """Dynamically generates or updates agent code using LLM."""
    def __init__(self):
        self.llm = llm_client
        self.version_file = ".digi/agent_versions.json"
        self.versions = self.load_versions()

    def load_versions(self):
        """Load agent versions from file."""
        if os.path.exists(self.version_file):
            try:
                with open(self.version_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load agent versions: {e}")
        return {}

    def save_versions(self):
        """Save agent versions to file."""
        try:
            with open(self.version_file, "w") as f:
                json.dump(self.versions, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save agent versions: {e}")

    def generate_agent_code(self, agent_name, features, client_id=None):
        """Generate or update agent code with specified features."""
        prompt = f"""
        Generate Python code for a DigiMan agent named '{agent_name}' with the following features:
        - {features}
        - Must include a class definition with at least 3 methods.
        - Must use log_action for logging.
        - Must support task execution via a run_task method.
        - Must integrate with LLMClient for intelligent task processing.
        - Must support client_id for multi-client operation.
        - Must check required API keys via audit_env_keys.
        - Must communicate with other agents via message_queue.
        - Must save metrics to METRICS_FILE.
        Current metrics: {json.dumps(metrics, indent=2)}
        """
        code = self.llm.process(prompt, max_tokens=1000)
        if "Error" in code:
            return None
        agent_dir = f".digi/clients/{client_id}" if client_id else ".digi"
        os.makedirs(agent_dir, exist_ok=True)
        filename = f"{agent_name.lower().replace(' ', '_')}_agent.py"
        filepath = os.path.join(agent_dir, filename)

        # Validate generated code
        score, reasons = evaluate_agent_quality(code)
        if score < 3:
            log_action("AgentBuilder", f"Failed to generate valid code for {agent_name}: {reasons}", client_id)
            return None

        try:
            with open(filepath, "w") as f:
                f.write(code)
            version_key = f"{agent_name}_{client_id}" if client_id else agent_name
            self.versions[version_key] = self.versions.get(version_key, 0) + 1
            self.save_versions()
            log_action("AgentBuilder", f"Generated agent {agent_name} version {self.versions[version_key]}", client_id)
            return code
        except Exception as e:
            logger.error(f"Failed to save agent code for {agent_name}: {e}")
            return None

    def inspect_agent(self, agent_name, client_id=None):
        """Inspect an agent's code and status."""
        agent_dir = f".digi/clients/{client_id}" if client_id else ".digi"
        filename = f"{agent_name.lower().replace(' ', '_')}_agent.py"
        filepath = os.path.join(agent_dir, filename)
        meta_path = os.path.join(agent_dir, f"{filename}.meta")
        status = "Not deployed"
        code_snippet = ""
        version = self.versions.get(f"{agent_name}_{client_id}" if client_id else agent_name, 0)

        if os.path.exists(filepath):
            try:
                with open(filepath, "r") as f:
                    code_snippet = f.read()[:500] + "..."  # First 500 chars
                with open(meta_path, "r") as f:
                    status = f.read().strip()
            except Exception as e:
                logger.error(f"Failed to inspect {agent_name}: {e}")
        return f"Agent: {agent_name}\nVersion: {version}\nStatus: {status}\nCode Snippet: {code_snippet}"

    def load_dynamic_agent(self, agent_name, code):
        """Load dynamically generated agent code."""
        module_name = f"dynamic_{agent_name.lower().replace(' ', '_')}"
        spec = importlib.util.spec_from_loader(module_name, loader=None)
        module = importlib.util.module_from_spec(spec)
        exec(code, module.__dict__)
        return getattr(module, agent_name.replace(" ", ""))

agent_builder = AgentBuilder()

class SubscriptionManager:
    """Manages client subscriptions and agent deployment."""
    def __init__(self):
        self.clients = {}
        self.subscriptions_file = ".digi/subscriptions.json"
        self.load_subscriptions()

    def load_subscriptions(self):
        """Load existing subscriptions from file."""
        if os.path.exists(self.subscriptions_file):
            try:
                with open(self.subscriptions_file, "r") as f:
                    self.clients = json.load(f)
            except Exception as e:
                logger.error(f"Failed to load subscriptions: {e}")

    def save_subscriptions(self):
        """Save subscriptions to file."""
        try:
            with open(self.subscriptions_file, "w") as f:
                json.dump(self.clients, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save subscriptions: {e}")

    def onboard_client(self, client_id, plan="basic"):
        """Onboard a new client and deploy agents."""
        if client_id not in self.clients:
            self.clients[client_id] = {"plan": plan, "agents_deployed": [], "created_at": str(datetime.datetime.now())}
            metrics["clients_onboarded"] += 1
            save_metrics()
            log_action("Subscription Manager", f"Onboarded client {client_id} with {plan} plan")
            update_task_queue("Client Onboarding Agent", {"task": f"Onboard client: {client_id}", "priority": 1, "dependent_on": []}, client_id)
            return f"Client {client_id} onboarded with {plan} plan. Onboarding agent deployed."
        return f"Client {client_id} already exists."

subscription_manager = SubscriptionManager()

# Agent Definitions
class ClientOnboardingAgent:
    """Manages client onboarding and agent deployment for each client."""
    def __init__(self, client_id=None):
        self.client_id = client_id
        self.llm = llm_client
        self.required_keys = ["HUBSPOT_API_KEY", "WEBFLOW_CLIENT_ID", "WEBFLOW_CLIENT_SECRET"]
        self.active = not audit_env_keys(self.required_keys)

    def run_task(self, task):
        task_description = task["task"] if isinstance(task, dict) else task
        log_action("Client Onboarding Agent", f"Running task: {task_description}", self.client_id)
        llm_response = self.llm.process(f"Client Onboarding Agent task: {task_description}", max_tokens=200)
        log_action("Client Onboarding Agent", f"LLM reasoning: {llm_response}", self.client_id)
        messages = message_queue.receive_messages("Client Onboarding Agent", self.client_id)
        for msg in messages:
            log_action("Client Onboarding Agent", f"Received message from {msg['sender']}: {msg['message']}", self.client_id)
        if "onboard client" in task_description.lower():
            client_id = re.search(r"client: (\w+)", task_description.lower())
            if client_id:
                self.onboard_client(client_id.group(1))

    def onboard_client(self, client_id):
        """Generate and deploy client-specific agent team."""
        agent_configs = self.llm.process(f"""
        Generate configurations for a team of AI agents for client {client_id} in the {business_phases[current_phase_index]} phase.
        Include: ManagerAgent, ScoutAgent, WebBuilderAgent, MarketingAgent, OutreachAgent, CloserAgent, CRMAgent, SocialsAgent, VisualsAgent, ContentAgent, SupportRetentionAgent, FinancialAllocationAgent, AnalystAgent, FranchiseBuilderAgent, FranchiseIntelligenceAgent, FranchiseRelationshipAgent, AutonomousSalesReplicator, MonetizationAgent.
        Each agent should have tailored tasks and API integrations (HubSpot, Webflow).
        """, max_tokens=1000)
        if "Error" in agent_configs:
            log_action("Client Onboarding Agent", f"Failed to generate agent configs: {agent_configs}", client_id)
            return
        try:
            configs = json.loads(agent_configs)
            for agent_name, features in configs.items():
                code = agent_builder.generate_agent_code(agent_name, features, client_id)
                if code:
                    AGENT_REGISTRY[agent_name] = agent_builder.load_dynamic_agent(agent_name, code)
                    deploy_agent(agent_name, AGENT_REGISTRY[agent_name], client_id)
                    subscription_manager.clients[client_id]["agents_deployed"].append(agent_name)
            subscription_manager.save_subscriptions()
            log_action("Client Onboarding Agent", f"Deployed agent team for client {client_id}", client_id)
        except Exception as e:
            logger.error(f"Failed to onboard client {client_id}: {e}")

class ManagerAgent:
    """Oversees all agents, coordinates tasks, and monitors performance."""
    def __init__(self, client_id=None):
        self.client_id = client_id
        self.llm = llm_client
        self.required_keys = ["HUBSPOT_API_KEY"]
        self.active = not audit_env_keys(self.required_keys)

    def run_task(self, task):
        task_description = task["task"] if isinstance(task, dict) else task
        log_action("Manager Agent", f"Running task: {task_description}", self.client_id)
        llm_response = self.llm.process(f"Manager Agent task: {task_description}", max_tokens=200)
        log_action("Manager Agent", f"LLM reasoning: {llm_response}", self.client_id)
        messages = message_queue.receive_messages("Manager Agent", self.client_id)
        for msg in messages:
            log_action("Manager Agent", f"Received message from {msg['sender']}: {msg['message']}", self.client_id)
        if "monitor" in task_description.lower():
            self.monitor_performance()
        elif "report" in task_description.lower():
            self.generate_report()
        elif "process command" in task_description.lower():
            self.process_command(task_description)
        elif "deploy" in task_description.lower():
            self.deploy_agent(task_description)
        elif "integrate" in task_description.lower():
            self.integrate_app(task_description)
        self.scale_self()

    def monitor_performance(self):
        if metrics["tasks_failed"] > 5:
            message_queue.send_message("Manager Agent", "Support Retention Agent", "Investigate high task failure rate", self.client_id)
            update_task_queue("Support Retention Agent", {"task": "Investigate task failures", "priority": 3, "dependent_on": []}, self.client_id)
        log_action("Manager Agent", f"Monitored performance: {metrics}", self.client_id)

    def generate_report(self):
        report = self.llm.process(f"Generate a performance report for client {self.client_id}. Metrics: {json.dumps(metrics, indent=2)}", max_tokens=500)
        log_action("Manager Agent", f"Generated report: {report}", self.client_id)
        message_queue.send_message("Manager Agent", "Analyst Agent", f"Analyze report: {report}", self.client_id)
        update_task_queue("Analyst Agent", {"task": f"Analyze report: {report}", "priority": 2, "dependent_on": ["Manager Agent"]}, self.client_id)

    def process_command(self, command):
        if "email" in command.lower():
            update_task_queue("Email Agent", {"task": "Process inbox", "priority": 2, "dependent_on": []}, self.client_id)
        elif "website" in command.lower():
            update_task_queue("Web Builder Agent", {"task": "Build website", "priority": 2, "dependent_on": []}, self.client_id)
        log_action("Manager Agent", f"Processed command: {command}", self.client_id)

    def deploy_agent(self, task):
        agent_name = re.search(r"deploy ([\w\s]+)", task.lower())
        if agent_name and agent_name.group(1).title() in AGENT_REGISTRY:
            deploy_agent(agent_name.group(1).title(), AGENT_REGISTRY[agent_name.group(1).title()], self.client_id)
            log_action("Manager Agent", f"Deployed {agent_name.group(1)}", self.client_id)
        else:
            log_action("Manager Agent", f"Invalid agent deployment request: {task}", self.client_id)

    def integrate_app(self, task):
        app_name = re.search(r"integrate (\w+)", task.lower())
        if app_name:
            log_action("Manager Agent", f"Integration queued for {app_name.group(1)}", self.client_id)

    def scale_self(self):
        if metrics["leads_generated"] > 1000:
            update_task_queue("Franchise Builder Agent", {"task": "Deploy new franchise", "priority": 3, "dependent_on": []}, self.client_id)
            log_action("Manager Agent", "Scaling: Initiated franchise deployment", self.client_id)

class EmailAgent:
    """Manages email communication, lead nurturing, and integrations."""
    def __init__(self, client_id=None):
        self.client_id = client_id
        self.llm = llm_client
        self.required_keys = ["EMAIL_ACCOUNT", "EMAIL_PASSWORD", "IMAP_SERVER", "SMTP_SERVER"]
        self.active = not audit_env_keys(self.required_keys)

    def run_task(self, task):
        task_description = task["task"] if isinstance(task, dict) else task
        log_action("Email Agent", f"Running task: {task_description}", self.client_id)
        llm_response = self.llm.process(f"Email Agent task: {task_description}", max_tokens=200)
        log_action("Email Agent", f"LLM reasoning: {llm_response}", self.client_id)
        messages = message_queue.receive_messages("Email Agent", self.client_id)
        for msg in messages:
            log_action("Email Agent", f"Received message from {msg['sender']}: {msg['message']}", self.client_id)
        if "process inbox" in task_description.lower():
            self.process_inbox()
        elif "send campaign" in task_description.lower():
            self.send_email_campaign()

    def process_inbox(self):
        history = set()
        try:
            if self.active:
                mail = imaplib.IMAP4_SSL(CONFIG["IMAP_SERVER"], CONFIG["IMAP_PORT"])
                mail.login(CONFIG["EMAIL_ACCOUNT"], CONFIG["EMAIL_PASSWORD"])
                mail.select("inbox")
                _, data = mail.search(None, "UNSEEN")
                for num in data[0].split():
                    _, data = mail.fetch(num, "(RFC822)")
                    msg = email.message_from_bytes(data[0][1])
                    subject = decode_header(msg["Subject"])[0][0]
                    subject = subject.decode() if isinstance(subject, bytes) else subject
                    sender = msg.get("From", "")
                    body = ""
                    for part in msg.walk():
                        if part.get_content_type() == "text/plain":
                            body += part.get_payload(decode=True).decode()
                    category = "Lead" if any(kw in subject.lower() for kw in ["inquiry", "quote", "interest"]) else "Support" if "help" in subject.lower() else "Unknown"
                    reply = self.llm.process(f"Craft a professional reply to email from {sender} with subject '{subject}': {body}", max_tokens=200)
                    smtp = smtplib.SMTP_SSL(CONFIG["SMTP_SERVER"], CONFIG["SMTP_PORT"])
                    smtp.login(CONFIG["EMAIL_ACCOUNT"], CONFIG["EMAIL_PASSWORD"])
                    response = EmailMessage()
                    response["Subject"] = f"Re: {subject}"
                    response["From"] = CONFIG["EMAIL_ACCOUNT"]
                    response["To"] = sender
                    response.set_content(reply)
                    smtp.send_message(response)
                    smtp.quit()
                    log_action("Email Agent", f"Replied to {sender}: {reply}", self.client_id)
                    if category == "Lead":
                        message_queue.send_message("Email Agent", "CRM Agent", f"Add lead: {sender}", self.client_id)
                        update_task_queue("CRM Agent", {"task": f"Add lead: {sender}", "priority": 2, "dependent_on": ["Email Agent"]}, self.client_id)
                        metrics["leads_generated"] += 1
                        save_metrics()
                    history.add(num.decode())
                mail.logout()
            else:
                log_action("Email Agent", "Email service disabled due to missing credentials", self.client_id)
                metrics["tasks_failed"] += 1
                save_metrics()
        except Exception as e:
            log_action("Email Agent", f"Inbox error: {e}", self.client_id)
            metrics["tasks_failed"] += 1
            save_metrics()

    def send_email_campaign(self):
        if self.active:
            leads = [{"email": f"lead{random.randint(1,100)}@example.com"} for _ in range(10)]
            campaign_content = self.llm.process("Generate engaging content for a marketing email campaign.", max_tokens=500)
            for lead in leads:
                smtp = smtplib.SMTP_SSL(CONFIG["SMTP_SERVER"], CONFIG["SMTP_PORT"])
                smtp.login(CONFIG["EMAIL_ACCOUNT"], CONFIG["EMAIL_PASSWORD"])
                msg = EmailMessage()
                msg["Subject"] = "Discover DigiMan Solutions"
                msg["From"] = CONFIG["EMAIL_ACCOUNT"]
                msg["To"] = lead["email"]
                msg.set_content(campaign_content)
                smtp.send_message(msg)
                smtp.quit()
                log_action("Email Agent", f"Sent campaign email to {lead['email']}", self.client_id)
        else:
            log_action("Email Agent", "Email campaign disabled due to missing credentials", self.client_id)

class WebBuilderAgent:
    """Builds and manages websites using Webflow API."""
    def __init__(self, client_id=None):
        self.client_id = client_id
        self.llm = llm_client
        self.required_keys = ["WEBFLOW_CLIENT_ID", "WEBFLOW_CLIENT_SECRET"]
        self.active = not audit_env_keys(self.required_keys)

    def run_task(self, task):
        task_description = task["task"] if isinstance(task, dict) else task
        log_action("Web Builder Agent", f"Running task: {task_description}", self.client_id)
        llm_response = self.llm.process(f"Web Builder Agent task: {task_description}", max_tokens=200)
        log_action("Web Builder Agent", f"LLM reasoning: {llm_response}", self.client_id)
        messages = message_queue.receive_messages("Web Builder Agent", self.client_id)
        for msg in messages:
            log_action("Web Builder Agent", f"Received message from {msg['sender']}: {msg['message']}", self.client_id)
        if "build website" in task_description.lower():
            business_type = re.search(r"website: (\w+)", task_description.lower())
            business_type = business_type.group(1) if business_type else "lead_capture"
            self.build_site(business_type)
        elif "optimize seo" in task_description.lower():
            self.optimize_seo()

    def build_site(self, business_type):
        domain = CONFIG.get("DOMAIN_NAME", "example.com")
        template = self.llm.process(f"Generate HTML template for a {business_type} website with domain {domain}", max_tokens=1000)
        if self.active:
            webflow_token = oauth_handler.get_access_token("webflow", self.client_id)
            if webflow_token:
                response = make_api_request(
                    "https://api.webflow.com/v2/sites",
                    method="POST",
                    headers={"Authorization": f"Bearer {webflow_token}"},
                    data={"name": f"{business_type}_{domain}", "template": template}
                )
                if response:
                    site_id = response.get("id")
                    log_action("Web Builder Agent", f"Built site {site_id} on Webflow for {business_type}", self.client_id)
                    message_queue.send_message("Web Builder Agent", "Marketing Agent", f"Promote site: {domain}", self.client_id)
                    update_task_queue("Marketing Agent", {"task": f"Promote site: {domain}", "priority": 2, "dependent_on": ["Web Builder Agent"]}, self.client_id)
                    return
        site_path = os.path.join(f".digi/clients/{self.client_id}", f"site_{business_type}.html")
        os.makedirs(os.path.dirname(site_path), exist_ok=True)
        try:
            with open(site_path, "w") as f:
                f.write(template)
            log_action("Web Builder Agent", f"Saved site for {business_type} at {site_path}", self.client_id)
            message_queue.send_message("Web Builder Agent", "Marketing Agent", f"Promote site: {domain}", self.client_id)
            update_task_queue("Marketing Agent", {"task": f"Promote site: {domain}", "priority": 2, "dependent_on": ["Web Builder Agent"]}, self.client_id)
        except Exception as e:
            logger.error(f"Failed to save site: {e}")

    def optimize_seo(self):
        seo_plan = self.llm.process("Generate an SEO optimization plan for a website.", max_tokens=500)
        log_action("Web Builder Agent", f"SEO optimized: {seo_plan}", self.client_id)

class PartnershipScoutAgent:
    """Identifies collaboration opportunities using search APIs."""
    def __init__(self, client_id=None):
        self.client_id = client_id
        self.llm = llm_client
        self.required_keys = [key for key in CONFIG if "SEARCH" in key.upper() or "SERP" in key.upper()]
        self.active = not audit_env_keys(self.required_keys)

    def run_task(self, task):
        task_description = task["task"] if isinstance(task, dict) else task
        log_action("Partnership Scout Agent", f"Running task: {task_description}", self.client_id)
        llm_response = self.llm.process(f"Partnership Scout Agent task: {task_description}", max_tokens=200)
        log_action("Partnership Scout Agent", f"LLM reasoning: {llm_response}", self.client_id)
        messages = message_queue.receive_messages("Partnership Scout Agent", self.client_id)
        for msg in messages:
            log_action("Partnership Scout Agent", f"Received message from {msg['sender']}: {msg['message']}", self.client_id)
        if "find partners" in task_description.lower():
            self.find_partners()

    def find_partners(self):
        if self.active and self.required_keys:
            for key in self.required_keys:
                api_key = CONFIG[key]
                endpoint = CONFIG.get(f"{key}_ENDPOINT", "https://api.search.com/v1/search")
                query = self.llm.process("Generate a search query for finding industry partners.", max_tokens=100)
                response = make_api_request(endpoint, api_key=api_key, data={"query": query})
                if response:
                    partners = response.get("results", [])
                    for partner in partners[:3]:
                        log_action("Partnership Scout Agent", f"Found partner: {partner.get('name')}", self.client_id)
                        message_queue.send_message("Partnership Scout Agent", "Outreach Agent", f"Contact partner: {partner.get('name')}", self.client_id)
                        update_task_queue("Outreach Agent", {"task": f"Contact partner: {partner.get('name')}", "priority": 2, "dependent_on": ["Partnership Scout Agent"]}, self.client_id)
                    metrics["leads_generated"] += len(partners[:3])
                    save_metrics()
                    return
        log_action("Partnership Scout Agent", "Search API disabled, no partners identified", self.client_id)

class ChainValidatorAgent:
    """Validates agent logic and workflow integrity."""
    def __init__(self, client_id=None):
        self.client_id = client_id
        self.llm = llm_client
        self.required_keys = []
        self.active = True

    def run_task(self, task):
        task_description = task["task"] if isinstance(task, dict) else task
        log_action("Chain Validator Agent", f"Running task: {task_description}", self.client_id)
        llm_response = self.llm.process(f"Chain Validator Agent task: {task_description}", max_tokens=200)
        log_action("Chain Validator Agent", f"LLM reasoning: {llm_response}", self.client_id)
        messages = message_queue.receive_messages("Chain Validator Agent", self.client_id)
        for msg in messages:
            log_action("Chain Validator Agent", f"Received message from {msg['sender']}: {msg['message']}", self.client_id)
        if "validate" in task_description.lower():
            self.validate_pipeline()

    def validate_pipeline(self):
        queue = load_task_queue(self.client_id)
        for agent_name, tasks in queue.items():
            if len(tasks) > 10:
                log_action("Chain Validator Agent", f"Warning: {agent_name} has {len(tasks)} pending tasks", self.client_id)
                message_queue.send_message("Chain Validator Agent", "Manager Agent", f"Task backlog for {agent_name}", self.client_id)
                send_message_to_digiman(f"Task backlog for {agent_name}", self.client_id)
        log_action("Chain Validator Agent", "Pipeline validated", self.client_id)

class StrategicPlannerAgent:
    """Plans agent deployment and resource allocation."""
    def __init__(self, client_id=None):
        self.client_id = client_id
        self.llm = llm_client
        self.required_keys = []
        self.active = True

    def run_task(self, task):
        task_description = task["task"] if isinstance(task, dict) else task
        log_action("Strategic Planner Agent", f"Running task: {task_description}", self.client_id)
        llm_response = self.llm.process(f"Strategic Planner Agent task: {task_description}", max_tokens=200)
        log_action("Strategic Planner Agent", f"LLM reasoning: {llm_response}", self.client_id)
        messages = message_queue.receive_messages("Strategic Planner Agent", self.client_id)
        for msg in messages:
            log_action("Strategic Planner Agent", f"Received message from {msg['sender']}: {msg['message']}", self.client_id)
        if "plan" in task_description.lower():
            self.plan_deployment()

    def plan_deployment(self):
        if metrics["leads_generated"] < 100:
            update_task_queue("Scout Agent", {"task": "Find niches", "priority": 3, "dependent_on": []}, self.client_id)
            update_task_queue("Outreach Agent", {"task": "Send outreach", "priority": 3, "dependent_on": []}, self.client_id)
        log_action("Strategic Planner Agent", "Deployment planned based on lead volume", self.client_id)

class CloserAgent:
    """Handles sales calls and deal closing."""
    def __init__(self, client_id=None):
        self.client_id = client_id
        self.llm = llm_client
        self.required_keys = [key for key in CONFIG if "CALL" in key.upper() or "VOICE" in key.upper()]
        self.active = not audit_env_keys(self.required_keys)

    def run_task(self, task):
        task_description = task["task"] if isinstance(task, dict) else task
        log_action("Closer Agent", f"Running task: {task_description}", self.client_id)
        llm_response = self.llm.process(f"Closer Agent task: {task_description}", max_tokens=200)
        log_action("Closer Agent", f"LLM reasoning: {llm_response}", self.client_id)
        messages = message_queue.receive_messages("Closer Agent", self.client_id)
        for msg in messages:
            log_action("Closer Agent", f"Received message from {msg['sender']}: {msg['message']}", self.client_id)
        if "close deal" in task_description.lower():
            self.close_deal()

    def close_deal(self):
        if self.active and self.required_keys:
            call_script = self.llm.process("Generate a sales call script for closing a deal.", max_tokens=500)
            log_action("Closer Agent", f"Initiated call with script: {call_script}", self.client_id)
            message_queue.send_message("Closer Agent", "CRM Agent", "Update deal status", self.client_id)
            update_task_queue("CRM Agent", {"task": "Update deal status", "priority": 2, "dependent_on": ["Closer Agent"]}, self.client_id)
        else:
            log_action("Closer Agent", "Call service disabled due to missing credentials", self.client_id)
            metrics["tasks_failed"] += 1
            save_metrics()

class CRMAgent:
    """Manages leads and client data in HubSpot."""
    def __init__(self, client_id=None):
        self.client_id = client_id
        self.llm = llm_client
        self.required_keys = ["HUBSPOT_API_KEY"]
        self.active = not audit_env_keys(self.required_keys)

    def run_task(self, task):
        task_description = task["task"] if isinstance(task, dict) else task
        log_action("CRM Agent", f"Running task: {task_description}", self.client_id)
        llm_response = self.llm.process(f"CRM Agent task: {task_description}", max_tokens=200)
        log_action("CRM Agent", f"LLM reasoning: {llm_response}", self.client_id)
        messages = message_queue.receive_messages("CRM Agent", self.client_id)
        for msg in messages:
            log_action("CRM Agent", f"Received message from {msg['sender']}: {msg['message']}", self.client_id)
            if "add lead" in msg["message"].lower():
                self.add_lead(msg["message"])
            elif "update deal" in msg["message"].lower():
                self.update_deal()
        if "add lead" in task_description.lower():
            self.add_lead(task_description)
        elif "update deal" in task_description.lower():
            self.update_deal()

    def add_lead(self, task):
        lead_email = re.search(r"[\w\.-]+@[\w\.-]+", task)
        if lead_email and self.active:
            hubspot_key = CONFIG.get("HUBSPOT_API_KEY")
            if hubspot_key:
                response = make_api_request(
                    "https://api.hubapi.com/crm/v3/objects/contacts",
                    method="POST",
                    api_key=hubspot_key,
                    data={"properties": {"email": lead_email.group()}}
                )
                if response:
                    log_action("CRM Agent", f"Added lead {lead_email.group()} to HubSpot", self.client_id)
                    return
            webflow_token = oauth_handler.get_access_token("hubspot", self.client_id)
            if webflow_token:
                response = make_api_request(
                    "https://api.hubapi.com/crm/v3/objects/contacts",
                    method="POST",
                    oauth_platform="hubspot",
                    data={"properties": {"email": lead_email.group()}},
                    client_id=self.client_id
                )
                if response:
                    log_action("CRM Agent", f"Added lead {lead_email.group()} to HubSpot via OAuth", self.client_id)
                    return
        log_action("CRM Agent", f"CRM service disabled, lead not added: {lead_email.group() if lead_email else 'unknown'}", self.client_id)

    def update_deal(self):
        deal_update = self.llm.process("Generate deal update details for CRM.", max_tokens=200)
        log_action("CRM Agent", f"Deal updated: {deal_update}", self.client_id)

class ScoutAgent:
    """Identifies market niches and target clients."""
    def __init__(self, client_id=None):
        self.client_id = client_id
        self.llm = llm_client
        self.required_keys = [key for key in CONFIG if "SEARCH" in key.upper() or "SERP" in key.upper()]
        self.active = not audit_env_keys(self.required_keys)

    def run_task(self, task):
        task_description = task["task"] if isinstance(task, dict) else task
        log_action("Scout Agent", f"Running task: {task_description}", self.client_id)
        llm_response = self.llm.process(f"Scout Agent task: {task_description}", max_tokens=200)
        log_action("Scout Agent", f"LLM reasoning: {llm_response}", self.client_id)
        messages = message_queue.receive_messages("Scout Agent", self.client_id)
        for msg in messages:
            log_action("Scout Agent", f"Received message from {msg['sender']}: {msg['message']}", self.client_id)
        if "find niches" in task_description.lower():
            self.find_niches()

    def find_niches(self):
        if self.active:
            niches = self.llm.process("Identify market niches for business growth.", max_tokens=200)
            log_action("Scout Agent", f"Niches identified: {niches}", self.client_id)
            message_queue.send_message("Scout Agent", "Marketing Agent", f"Create campaign for niches: {niches}", self.client_id)
            update_task_queue("Marketing Agent", {"task": f"Create campaign for tech niche", "priority": 2, "dependent_on": ["Scout Agent"]}, self.client_id)
        else:
            log_action("Scout Agent", "Search API disabled, no niches identified", self.client_id)

class BrandManagerAgent:
    """Coordinates branding and visual consistency."""
    def __init__(self, client_id=None):
        self.client_id = client_id
        self.llm = llm_client
        self.required_keys = []
        self.active = True

    def run_task(self, task):
        task_description = task["task"] if isinstance(task, dict) else task
        log_action("Brand Manager Agent", f"Running task: {task_description}", self.client_id)
        llm_response = self.llm.process(f"Brand Manager Agent task: {task_description}", max_tokens=200)
        log_action("Brand Manager Agent", f"LLM reasoning: {llm_response}", self.client_id)
        messages = message_queue.receive_messages("Brand Manager Agent", self.client_id)
        for msg in messages:
            log_action("Brand Manager Agent", f"Received message from {msg['sender']}: {msg['message']}", self.client_id)
        if "manage branding" in task_description.lower():
            self.manage_branding()

    def manage_branding(self):
        branding_guidelines = self.llm.process("Generate branding guidelines.", max_tokens=500)
        log_action("Brand Manager Agent", f"Branding ensured: {branding_guidelines}", self.client_id)
        message_queue.send_message("Brand Manager Agent", "Visuals Agent", "Create branding assets", self.client_id)
        update_task_queue("Visuals Agent", {"task": "Create branding assets", "priority": 2, "dependent_on": ["Brand Manager Agent"]}, self.client_id)

class MarketingAgent:
    """Runs marketing campaigns and content creation."""
    def __init__(self, client_id=None):
        self.client_id = client_id
        self.llm = llm_client
        self.required_keys = [key for key in CONFIG if "MARKETING" in key.upper() or "ADS" in key.upper()]
        self.active = not audit_env_keys(self.required_keys)

    def run_task(self, task):
        task_description = task["task"] if isinstance(task, dict) else task
        log_action("Marketing Agent", f"Running task: {task_description}", self.client_id)
        llm_response = self.llm.process(f"Marketing Agent task: {task_description}", max_tokens=200)
        log_action("Marketing Agent", f"LLM reasoning: {llm_response}", self.client_id)
        messages = message_queue.receive_messages("Marketing Agent", self.client_id)
        for msg in messages:
            log_action("Marketing Agent", f"Received message from {msg['sender']}: {msg['message']}", self.client_id)
            if "promote site" in msg["message"].lower():
                self.promote_site()
            elif "create campaign" in msg["message"].lower():
                self.create_campaign(msg["message"])
        if "create campaign" in task_description.lower():
            self.create_campaign(task_description)
        elif "promote site" in task_description.lower():
            self.promote_site()

    def create_campaign(self, task):
        niche = re.search(r"for (\w+) niche", task.lower())
        niche = niche.group(1) if niche else "general"
        campaign_plan = self.llm.process(f"Generate a marketing campaign plan for {niche} niche.", max_tokens=500)
        if self.active:
            log_action("Marketing Agent", f"Launched campaign for {niche}: {campaign_plan}", self.client_id)
            message_queue.send_message("Marketing Agent", "Socials Agent", f"Post campaign content for {niche}", self.client_id)
            update_task_queue("Socials Agent", {"task": f"Post campaign content for {niche}", "priority": 2, "dependent_on": ["Marketing Agent"]}, self.client_id)
        else:
            log_action("Marketing Agent", f"Campaign disabled due to missing credentials: {niche}", self.client_id)

    def promote_site(self):
        promotion_strategy = self.llm.process("Generate a site promotion strategy.", max_tokens=200)
        log_action("Marketing Agent", f"Site promoted: {promotion_strategy}", self.client_id)

class VisualsAgent:
    """Designs visual assets for branding and marketing."""
    def __init__(self, client_id=None):
        self.client_id = client_id
        self.llm = llm_client
        self.required_keys = [key for key in CONFIG if "DESIGN" in key.upper() or "CANVA" in key.upper()]
        self.active = not audit_env_keys(self.required_keys)

    def run_task(self, task):
        task_description = task["task"] if isinstance(task, dict) else task
        log_action("Visuals Agent", f"Running task: {task_description}", self.client_id)
        llm_response = self.llm.process(f"Visuals Agent task: {task_description}", max_tokens=200)
        log_action("Visuals Agent", f"LLM reasoning: {llm_response}", self.client_id)
        messages = message_queue.receive_messages("Visuals Agent", self.client_id)
        for msg in messages:
            log_action("Visuals Agent", f"Received message from {msg['sender']}: {msg['message']}", self.client_id)
        if "create branding assets" in task_description.lower():
            self.create_branding_assets()

    def create_branding_assets(self):
        if self.active:
            asset_description = self.llm.process("Describe branding assets to create.", max_tokens=200)
            log_action("Visuals Agent", f"Created assets: {asset_description}", self.client_id)
        else:
            log_action("Visuals Agent", "Design API disabled, no assets created", self.client_id)

class SocialsAgent:
    """Manages social media presence and engagement."""
    def __init__(self, client_id=None):
        self.client_id = client_id
        self.llm = llm_client
        self.required_keys = [key for key in CONFIG if "SOCIAL" in key.upper() or "INSTAGRAM" in key.upper() or "TWITTER" in key.upper()]
        self.active = not audit_env_keys(self.required_keys)

    def run_task(self, task):
        task_description = task["task"] if isinstance(task, dict) else task
        log_action("Socials Agent", f"Running task: {task_description}", self.client_id)
        llm_response = self.llm.process(f"Socials Agent task: {task_description}", max_tokens=200)
        log_action("Socials Agent", f"LLM reasoning: {llm_response}", self.client_id)
        messages = message_queue.receive_messages("Socials Agent", self.client_id)
        for msg in messages:
            log_action("Socials Agent", f"Received message from {msg['sender']}: {msg['message']}", self.client_id)
        if "post content" in task_description.lower():
            self.post_content()

    def post_content(self):
        if self.active:
            post_content = self.llm.process("Generate social media post content.", max_tokens=200)
            log_action("Socials Agent", f"Posted content: {post_content}", self.client_id)
            metrics["leads_generated"] += 20
            save_metrics()
        else:
            log_action("Socials Agent", "Social API disabled, no content posted", self.client_id)

class OutreachAgent:
    """Handles outbound communication and lead generation."""
    def __init__(self, client_id=None):
        self.client_id = client_id
        self.llm = llm_client
        self.required_keys = ["EMAIL_ACCOUNT", "EMAIL_PASSWORD"]
        self.active = not audit_env_keys(self.required_keys)

    def run_task(self, task):
        task_description = task["task"] if isinstance(task, dict) else task
        log_action("Outreach Agent", f"Running task: {task_description}", self.client_id)
        llm_response = self.llm.process(f"Outreach Agent task: {task_description}", max_tokens=200)
        log_action("Outreach Agent", f"LLM reasoning: {llm_response}", self.client_id)
        messages = message_queue.receive_messages("Outreach Agent", self.client_id)
        for msg in messages:
            log_action("Outreach Agent", f"Received message from {msg['sender']}: {msg['message']}", self.client_id)
        if "contact partner" in task_description.lower():
            self.contact_partner(task_description)

    def contact_partner(self, task):
        partner = re.search(r"partner: (.+)", task)
        if partner and self.active:
            smtp = smtplib.SMTP_SSL(CONFIG["SMTP_SERVER"], CONFIG["SMTP_PORT"])
            smtp.login(CONFIG["EMAIL_ACCOUNT"], CONFIG["EMAIL_PASSWORD"])
            msg = EmailMessage()
            msg["Subject"] = "Partnership Opportunity"
            msg["From"] = CONFIG["EMAIL_ACCOUNT"]
            msg["To"] = partner.group(1)
            msg.set_content(self.llm.process(f"Generate a partnership outreach email for {partner.group(1)}.", max_tokens=200))
            smtp.send_message(msg)
            smtp.quit()
            log_action("Outreach Agent", f"Contacted partner: {partner.group(1)}", self.client_id)
        else:
            log_action("Outreach Agent", f"Outreach disabled, no contact sent to {partner.group(1) if partner else 'unknown'}", self.client_id)

class Outreach_Agent:
    """Handles outbound communication and lead generation."""
    def __init__(self, client_id=None):
        self.client_id = client_id
        self.llm = llm_client
        self.required_keys = ["EMAIL_ACCOUNT", "EMAIL_PASSWORD"]
        self.active = not audit_env_keys(self.required_keys)

    def run_task(self, task):
        task_description = task["task"] if isinstance(task, dict) else task
        log_action("Outreach Agent", f"Running task: {task_description}", self.client_id)
        llm_response = self.llm.process(f"Outreach Agent task: {task_description}", max_tokens=200)
        log_action("Outreach Agent", f"LLM reasoning: {llm_response}", self.client_id)
        messages = message_queue.receive_messages("Outreach Agent", self.client_id)
        for msg in messages:
            log_action("Outreach Agent", f"Received message from {msg['sender']}: {msg['message']}", self.client_id)
        if "contact partner" in task_description.lower():
            self.contact_partner(task_description)

    def contact_partner(self, task):
        partner = re.search(r"partner: (.+)", task)
        if partner and self.active:
            smtp = smtplib.SMTP_SSL(CONFIG["SMTP_SERVER"], CONFIG["SMTP_PORT"])
            smtp.login(CONFIG["EMAIL_ACCOUNT"], CONFIG["EMAIL_PASSWORD"])
            msg = EmailMessage()
            msg["Subject"] = "Partnership Opportunity"
            msg["From"] = CONFIG["EMAIL_ACCOUNT"]
            msg["To"] = partner.group(1)
            msg.set_content(self.llm.process(f"Generate a partnership outreach email for {partner.group(1)}.", max_tokens=200))
            smtp.send_message(msg)
            smtp.quit()
            log_action("Outreach Agent", f"Contacted partner: {partner.group(1)}", self.client_id)
        else:
            log_action("Outreach Agent", f"Outreach disabled, no contact sent to {partner.group(1) if partner else 'unknown'}", self.client_id)

class SubscriptionAgent:
    """Manages billing and subscription plans."""
    def __init__(self, client_id=None):
        self.client_id = client_id
        self.llm = llm_client
        self.required_keys = [key for key in CONFIG if "PAYMENT" in key.upper() or "STRIPE" in key.upper()]
        self.active = not audit_env_keys(self.required_keys)

    def run_task(self, task):
        task_description = task["task"] if isinstance(task, dict) else task
        log_action("Subscription Agent", f"Running task: {task_description}", self.client_id)
        llm_response = self.llm.process(f"Subscription Agent task: {task_description}", max_tokens=200)
        log_action("Subscription Agent", f"LLM reasoning: {llm_response}", self.client_id)
        messages = message_queue.receive_messages("Subscription Agent", self.client_id)
        for msg in messages:
            log_action("Subscription Agent", f"Received message from {msg['sender']}: {msg['message']}", self.client_id)
        if "manage billing" in task_description.lower():
            self.manage_billing()

    def manage_billing(self):
        if self.active:
            billing_details = self.llm.process("Generate billing details for client subscription.", max_tokens=200)
            log_action("Subscription Agent", f"Processed billing: {billing_details}", self.client_id)
        else:
            log_action("Subscription Agent", "Billing disabled due to missing credentials", self.client_id)

class SupportRetentionAgent:
    """Handles customer support tickets and retention strategies."""
    def __init__(self, client_id=None):
        self.client_id = client_id
        self.llm = llm_client
        self.required_keys = [key for key in CONFIG if "SUPPORT" in key.upper() or "ZENDESK" in key.upper()]
        self.active = not audit_env_keys(self.required_keys)

    def run_task(self, task):
        task_description = task["task"] if isinstance(task, dict) else task
        log_action("Support Retention Agent", f"Running task: {task_description}", self.client_id)
        llm_response = self.llm.process(f"Support Retention Agent task: {task_description}", max_tokens=200)
        log_action("Support Retention Agent", f"LLM reasoning: {llm_response}", self.client_id)
        messages = message_queue.receive_messages("Support Retention Agent", self.client_id)
        for msg in messages:
            log_action("Support Retention Agent", f"Received message from {msg['sender']}: {msg['message']}", self.client_id)
        if "handle ticket" in task_description.lower() or "investigate" in task_description.lower():
            self.handle_ticket()
        elif "reduce churn" in task_description.lower():
            self.reduce_churn()

    def handle_ticket(self):
        if self.active:
            ticket_response = self.llm.process("Generate a response for a customer support ticket.", max_tokens=200)
            log_action("Support Retention Agent", f"Resolved ticket: {ticket_response}", self.client_id)
        else:
            log_action("Support Retention Agent", "Support disabled due to missing credentials", self.client_id)

    def reduce_churn(self):
        retention_strategy = self.llm.process("Generate a retention strategy to reduce churn.", max_tokens=200)
        log_action("Support Retention Agent", f"Applied retention strategy: {retention_strategy}", self.client_id)
        message_queue.send_message("Support Retention Agent", "Email Agent", "Send retention email campaign", self.client_id)
        update_task_queue("Email Agent", {"task": "Send retention email campaign", "priority": 2, "dependent_on": ["Support Retention Agent"]}, self.client_id)

class FinancialAllocationAgent:
    """Manages budgets, payments, and financial performance."""
    def __init__(self, client_id=None):
        self.client_id = client_id
        self.llm = llm_client
        self.required_keys = [key for key in CONFIG if "PAYMENT" in key.upper() or "STRIPE" in key.upper()]
        self.active = not audit_env_keys(self.required_keys)

    def run_task(self, task):
        task_description = task["task"] if isinstance(task, dict) else task
        log_action("Financial Allocation Agent", f"Running task: {task_description}", self.client_id)
        llm_response = self.llm.process(f"Financial Allocation Agent task: {task_description}", max_tokens=200)
        log_action("Financial Allocation Agent", f"LLM reasoning: {llm_response}", self.client_id)
        messages = message_queue.receive_messages("Financial Allocation Agent", self.client_id)
        for msg in messages:
            log_action("Financial Allocation Agent", f"Received message from {msg['sender']}: {msg['message']}", self.client_id)
        if "manage budget" in task_description.lower():
            self.manage_budget()

    def manage_budget(self):
        if self.active:
            budget_plan = self.llm.process("Generate a budget allocation plan.", max_tokens=200)
            log_action("Financial Allocation Agent", f"Managed budget: {budget_plan}", self.client_id)
        else:
            log_action("Financial Allocation Agent", "Budget management disabled due to missing credentials", self.client_id)

class ContentAgent:
    """Creates content for blogs, scripts, and product descriptions."""
    def __init__(self, client_id=None):
        self.client_id = client_id
        self.llm = llm_client
        self.required_keys = []
        self.active = True

    def run_task(self, task):
        task_description = task["task"] if isinstance(task, dict) else task
        log_action("Content Agent", f"Running task: {task_description}", self.client_id)
        llm_response = self.llm.process(f"Content Agent task: {task_description}", max_tokens=200)
        log_action("Content Agent", f"LLM reasoning: {llm_response}", self.client_id)
        messages = message_queue.receive_messages("Content Agent", self.client_id)
        for msg in messages:
            log_action("Content Agent", f"Received message from {msg['sender']}: {msg['message']}", self.client_id)
        if "write content" in task_description.lower():
            self.write_content()

    def write_content(self):
        content = self.llm.process("Generate a blog post or product description.", max_tokens=500)
        log_action("Content Agent", f"Created content: {content}", self.client_id)
        message_queue.send_message("Content Agent", "Socials Agent", f"Post content: {content}", self.client_id)
        update_task_queue("Socials Agent", {"task": f"Post content: {content}", "priority": 2, "dependent_on": ["Content Agent"]}, self.client_id)

class AnalystAgent:
    """Analyzes business metrics and trends."""
    def __init__(self, client_id=None):
        self.client_id = client_id
        self.llm = llm_client
        self.required_keys = [key for key in CONFIG if "ANALYTICS" in key.upper()]
        self.active = not audit_env_keys(self.required_keys)

    def run_task(self, task):
        task_description = task["task"] if isinstance(task, dict) else task
        log_action("Analyst Agent", f"Running task: {task_description}", self.client_id)
        llm_response = self.llm.process(f"Analyst Agent task: {task_description}", max_tokens=200)
        log_action("Analyst Agent", f"LLM reasoning: {llm_response}", self.client_id)
        messages = message_queue.receive_messages("Analyst Agent", self.client_id)
        for msg in messages:
            log_action("Analyst Agent", f"Received message from {msg['sender']}: {msg['message']}", self.client_id)
        if "analyze" in task_description.lower():
            self.analyze_data(task_description)

    def analyze_data(self, task):
        analysis = self.llm.process(f"Analyze data for client {self.client_id}: {task}. Metrics: {json.dumps(metrics, indent=2)}", max_tokens=200)
        log_action("Analyst Agent", f"Analyzed data: {analysis}", self.client_id)
        if "report" in task.lower():
            message_queue.send_message("Analyst Agent", "Manager Agent", f"Review analytics: {analysis}", self.client_id)
            update_task_queue("Manager Agent", {"task": f"Review analytics report: {analysis}", "priority": 2, "dependent_on": ["Analyst Agent"]}, self.client_id)

class FranchiseBuilderAgent:
    """Clones and deploys DigiMan franchises."""
    def __init__(self, client_id=None):
        self.client_id = client_id
        self.llm = llm_client
        self.required_keys = []
        self.active = True

    def run_task(self, task):
        task_description = task["task"] if isinstance(task, dict) else task
        log_action("Franchise Builder Agent", f"Running task: {task_description}", self.client_id)
        llm_response = self.llm.process(f"Franchise Builder Agent task: {task_description}", max_tokens=200)
        log_action("Franchise Builder Agent", f"LLM reasoning: {llm_response}", self.client_id)
        messages = message_queue.receive_messages("Franchise Builder Agent", self.client_id)
        for msg in messages:
            log_action("Franchise Builder Agent", f"Received message from {msg['sender']}: {msg['message']}", self.client_id)
        if "deploy franchise" in task_description.lower():
            self.deploy_franchise()

    def deploy_franchise(self):
        franchise_id = f"franchise_{random.randint(1000,9999)}"
        log_action("Franchise Builder Agent", f"Deployed franchise: {franchise_id}", self.client_id)
        message_queue.send_message("Franchise Builder Agent", "Franchise Relationship Agent", f"Onboard franchise: {franchise_id}", self.client_id)
        update_task_queue("Franchise Relationship Agent", {"task": f"Onboard franchise: {franchise_id}", "priority": 2, "dependent_on": ["Franchise Builder Agent"]}, self.client_id)

class FranchiseIntelligenceAgent:
    """Analyzes franchise performance and optimizes agents."""
    def __init__(self, client_id=None):
        self.client_id = client_id
        self.llm = llm_client
        self.required_keys = []
        self.active = True

    def run_task(self, task):
        task_description = task["task"] if isinstance(task, dict) else task
        log_action("Franchise Intelligence Agent", f"Running task: {task_description}", self.client_id)
        llm_response = self.llm.process(f"Franchise Intelligence Agent task: {task_description}", max_tokens=200)
        log_action("Franchise Intelligence Agent", f"LLM reasoning: {llm_response}", self.client_id)
        messages = message_queue.receive_messages("Franchise Intelligence Agent", self.client_id)
        for msg in messages:
            log_action("Franchise Intelligence Agent", f"Received message from {msg['sender']}: {msg['message']}", self.client_id)
        if "analyze franchise" in task_description.lower():
            self.analyze_franchise()

    def analyze_franchise(self):
        analysis = self.llm.process(f"Analyze franchise performance for client {self.client_id}.", max_tokens=200)
        log_action("Franchise Intelligence Agent", f"Franchise analysis: {analysis}", self.client_id)

class FranchiseRelationshipAgent:
    """Supports franchise operators with onboarding and training."""
    def __init__(self, client_id=None):
        self.client_id = client_id
        self.llm = llm_client
        self.required_keys = []
        self.active = True

    def run_task(self, task):
        task_description = task["task"] if isinstance(task, dict) else task
        log_action("Franchise Relationship Agent", f"Running task: {task_description}", self.client_id)
        llm_response = self.llm.process(f"Franchise Relationship Agent task: {task_description}", max_tokens=200)
        log_action("Franchise Relationship Agent", f"LLM reasoning: {llm_response}", self.client_id)
        messages = message_queue.receive_messages("Franchise Relationship Agent", self.client_id)
        for msg in messages:
            log_action("Franchise Relationship Agent", f"Received message from {msg['sender']}: {msg['message']}", self.client_id)
        if "onboard franchise" in task_description.lower():
            self.onboard_franchise(task_description)

    def onboard_franchise(self, task):
        franchise_id = re.search(r"franchise: (\w+)", task)
        onboarding_plan = self.llm.process(f"Onboard franchise {franchise_id.group(1) if franchise_id else 'unknown'} for client {self.client_id}.", max_tokens=200)
        log_action("Franchise Relationship Agent", f"Onboarded franchise: {franchise_id.group(1) if franchise_id else 'unknown'}. Plan: {onboarding_plan}", self.client_id)

class AutonomousSalesReplicator:
    """Replicates successful sales strategies across niches."""
    def __init__(self, client_id=None):
        self.client_id = client_id
        self.llm = llm_client
        self.required_keys = []
        self.active = True

    def run_task(self, task):
        task_description = task["task"] if isinstance(task, dict) else task
        log_action("Autonomous Sales Replicator", f"Running task: {task_description}", self.client_id)
        llm_response = self.llm.process(f"Autonomous Sales Replicator task: {task_description}", max_tokens=200)
        log_action("Autonomous Sales Replicator", f"LLM reasoning: {llm_response}", self.client_id)
        messages = message_queue.receive_messages("Autonomous Sales Replicator", self.client_id)
        for msg in messages:
            log_action("Autonomous Sales Replicator", f"Received message from {msg['sender']}: {msg['message']}", self.client_id)
        if "replicate strategy" in task_description.lower():
            self.replicate_strategy()

    def replicate_strategy(self):
        strategy = self.llm.process("Replicate a successful sales strategy.", max_tokens=200)
        log_action("Autonomous Sales Replicator", f"Replicated sales strategy: {strategy}", self.client_id)
        message_queue.send_message("Autonomous Sales Replicator", "Closer Agent", f"Close deal with replicated strategy: {strategy}", self.client_id)
        update_task_queue("Closer Agent", {"task": f"Close deal with replicated strategy", "priority": 2, "dependent_on": ["Autonomous Sales Replicator"]}, self.client_id)

class MonetizationAgent:
    """Sets pricing strategy and revenue optimization (internal use only)."""
    def __init__(self, client_id=None):
        self.client_id = client_id
        self.llm = llm_client
        self.required_keys = []
        self.active = True

    def run_task(self, task):
        task_description = task["task"] if isinstance(task, dict) else task
        log_action("Monetization Agent", f"Running task: {task_description}", self.client_id)
        llm_response = self.llm.process(f"Monetization Agent task: {task_description}", max_tokens=200)
        log_action("Monetization Agent", f"LLM reasoning: {llm_response}", self.client_id)
        messages = message_queue.receive_messages("Monetization Agent", self.client_id)
        for msg in messages:
            log_action("Monetization Agent", f"Received message from {msg['sender']}: {msg['message']}", self.client_id)
        if "optimize pricing" in task_description.lower():
            self.optimize_pricing()

    def optimize_pricing(self):
        pricing_strategy = self.llm.process("Generate a pricing strategy for internal revenue optimization.", max_tokens=200)
        log_action("Monetization Agent", f"Optimized pricing: {pricing_strategy}", self.client_id)

# Agent Registry
AGENT_REGISTRY = {
    "Client Onboarding Agent": ClientOnboardingAgent,
    "Manager Agent": ManagerAgent,
    "Email Agent": EmailAgent,
    "Web Builder Agent": WebBuilderAgent,
    "Partnership Scout Agent": PartnershipScoutAgent,
    "Chain Validator Agent": ChainValidatorAgent,
    "Strategic Planner Agent": StrategicPlannerAgent,
    "Closer Agent": CloserAgent,
    "CRM Agent": CRMAgent,
    "Scout Agent": ScoutAgent,
    "Brand Manager Agent": BrandManagerAgent,
    "Marketing Agent": MarketingAgent,
    "Visuals Agent": VisualsAgent,
    "Socials Agent": SocialsAgent,
    "Outreach Agent": OutreachAgent,
    "Subscription Agent": SubscriptionAgent,
    "Support Retention Agent": SupportRetentionAgent,
    "Financial Allocation Agent": FinancialAllocationAgent,
    "Content Agent": ContentAgent,
    "Analyst Agent": AnalystAgent,
    "Franchise Builder Agent": FranchiseBuilderAgent,
    "Franchise Intelligence Agent": FranchiseIntelligenceAgent,
    "Franchise Relationship Agent": FranchiseRelationshipAgent,
    "Autonomous Sales Replicator": AutonomousSalesReplicator,
    "Monetization Agent": MonetizationAgent
}

# Agent Deployment
def deploy_agent(agent_name, agent_class, client_id=None):
    """Deploy agent by writing its code to file and scoring it."""
    agent_dir = f".digi/clients/{client_id}" if client_id else ".digi"
    os.makedirs(agent_dir, exist_ok=True)
    filename = f"{agent_name.lower().replace(' ', '_')}_agent.py"
    filepath = os.path.join(agent_dir, filename)
    meta_path = os.path.join(agent_dir, f"{filename}.meta")
    scoreboard_path = os.path.join(agent_dir, "scoreboard.txt")

    code = inspect.getsource(agent_class)
    score, reasons = evaluate_agent_quality(code)

    try:
        with open(filepath, "w") as f:
            f.write(code)
        with open(meta_path, "w") as meta:
            meta.write(f"Score: {score}/4\n")
            for reason in reasons:
                meta.write(f"Issue: {reason}\n")
            if score < 3:
                meta.write("LOCKED\n")
                log_action(agent_name, "Agent locked due to low score", client_id)
            else:
                meta.write("DEPLOYED\n")
        with open(scoreboard_path, "a") as board:
            board.write(f"{agent_name}: {score}/4 - {' | '.join(reasons) if reasons else 'OK'}\n")
    except Exception as e:
        logger.error(f"Failed to deploy {agent_name}: {e}")
        return False

    if score >= 3:
        metrics["agents_generated"] += 1
        log_action(agent_name, f"Deployed with score: {score}/4", client_id)
        return True
    return False

def run_agents(client_id=None):
    """Execute all deployed agents' tasks."""
    queue = load_task_queue(client_id)
    for agent_name, agent_class in AGENT_REGISTRY.items():
        agent_instance = agent_class(client_id)
        tasks = sorted(queue.get(agent_name, []), key=lambda x: x.get("priority", 1), reverse=True)
        for task in tasks:
            dependencies_met = all(dep in [t["sender"] for t in message_queue.receive_messages(agent_name, client_id)] for dep in task.get("dependent_on", []))
            if dependencies_met:
                try:
                    agent_instance.run_task(task)
                except Exception as e:
                    log_action(agent_name, f"Task error: {e}", client_id)
                    metrics["tasks_failed"] += 1
                    save_metrics()
        queue[agent_name] = []
    queue_path = os.path.join(f".digi/clients/{client_id}" if client_id else ".digi", "agent_queue.json")
    try:
        with open(queue_path, "w") as f:
            json.dump(queue, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save task queue: {e}")

# Autonomous Loop
def autonomous_loop():
    global current_phase_index
    agent_tasks = [
        ("Chain Validator Agent", "Validates logic and system alignment"),
        ("Strategic Planner Agent", "Deploys agents based on business needs"),
        ("Closer Agent", "Handles calls, objections, and closes deals"),
        ("CRM Agent", "Captures and manages lead intelligence"),
        ("Scout Agent", "Finds niches, pain points, and target clients"),
        ("Brand Manager Agent", "Assigns visuals, socials, and marketing"),
        ("Marketing Agent", "Writes email/DMs, launches funnels"),
        ("Visuals Agent", "Designs branding assets"),
        ("Socials Agent", "Posts content, grows channels"),
        ("Outreach Agent", "Sends outbound emails and messages"),
        ("Subscription Agent", "Manages billing and upgrades"),
        ("Support Retention Agent", "Handles tickets and feedback"),
        ("Financial Allocation Agent", "Manages budgets and payments"),
        ("Content Agent", "Creates blog posts and descriptions"),
        ("Analyst Agent", "Reviews pricing, growth, and trends"),
        ("Franchise Builder Agent", "Clones and deploys DigiMan franchises"),
        ("Franchise Intelligence Agent", "Analyzes franchise performance"),
        ("Franchise Relationship Agent", "Handles onboarding and support for franchises"),
        ("Autonomous Sales Replicator", "Clones high-converting sales strategies"),
        ("Email Agent", "Manages inbound and outbound client communication"),
        ("Partnership Scout Agent", "Identifies collaboration opportunities"),
        ("Monetization Agent", "Optimizes internal pricing strategies")
    ]

    required_keys = set(key for agent_cls in AGENT_REGISTRY.values() for key in agent_cls(None).required_keys)
    missing_keys = audit_env_keys(required_keys)
    if missing_keys:
        log_action("DigiMan Core", f"Cannot run autonomous loop due to missing keys: {', '.join(missing_keys)}")
        return

    overrides = check_owner_overrides()
    for agent_name, _ in agent_tasks:
        if agent_name not in overrides:
            deploy_agent(agent_name, AGENT_REGISTRY[agent_name])
    for agent_name, task_desc in agent_tasks:
        update_task_queue(agent_name, {"task": task_desc, "priority": 1, "dependent_on": []})
    run_agents()
    current_phase_index = (current_phase_index + 1) % len(business_phases)

# Interactive CLI Loop
def cli_loop():
    """Interactive CLI for DigiMan with LLM-driven responses."""
    print("DigiMan AI Chat Interface - Type 'help' for commands, 'exit' to quit")
    missing_keys = audit_env_keys(["XAI_API_KEY", "HUBSPOT_API_KEY", "WEBFLOW_CLIENT_ID", "WEBFLOW_CLIENT_SECRET"])
    if missing_keys:
        print(f"Warning: Missing API keys: {', '.join(missing_keys)}. Some features may be disabled.")
    while True:
        try:
            user_input = input("You: ").strip()
            if user_input.lower() == "exit":
                print("Exiting chat interface...")
                break
            response = chat_interface.process_input(user_input)
            print(f"DigiMan: {response}")
        except KeyboardInterrupt:
            print("\nExiting chat interface...")
            break
        except Exception as e:
            logger.error(f"CLI error: {e}")
            print("DigiMan: An error occurred. Please try again.")

# Streamlit App
def streamlit_app():
    """Streamlit wrapper for Digiman AI – Smart, conversational COO app."""
    st.set_page_config(page_title="Digiman AI", layout="wide")
    st.title("Digiman AI – Your Autonomous COO")
    st.write("Talk to your COO, launch agents, integrate apps, and track growth from your phone or desktop.")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.subheader("Business Dashboard")
    metrics_df = pd.DataFrame([metrics])
    st.bar_chart(metrics_df[["leads_generated", "revenue_generated", "tasks_processed"]])
    st.json(metrics)

    st.subheader("Integrate Apps")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Integrate HubSpot"):
            auth_url, _ = oauth_handler.get_oauth_session("hubspot").create_authorization_url(oauth_handler.platforms["hubspot"]["auth_url"])
            st.write(f"Click [here]({auth_url}) to integrate HubSpot.")
    with col2:
        if st.button("Integrate Webflow"):
            auth_url, _ = oauth_handler.get_oauth_session("webflow").create_authorization_url(oauth_handler.platforms["webflow"]["auth_url"])
            st.write(f"Click [here]({auth_url}) to integrate Webflow.")

    st.subheader("Launch Agents")
    cols = st.columns(4)
    for i, agent_name in enumerate(AGENT_REGISTRY.keys()):
        with cols[i % 4]:
            if st.button(f"Deploy {agent_name}"):
                response = chat_interface.process_input(f"deploy {agent_name}")
                st.session_state.chat_history.append({"user": f"deploy {agent_name}", "digiman": response})
                st.write(response)

    st.subheader("Onboard Client")
    client_id = st.text_input("Client ID")
    if st.button("Onboard Client"):
        response = chat_interface.process_input(f"onboard client {client_id}")
        st.session_state.chat_history.append({"user": f"onboard client {client_id}", "digiman": response})
        st.write(response)

    st.subheader("Talk to Your COO")
    user_input = st.text_input("Ask about your business or give a command:", key="input")
    if user_input:
        response = chat_interface.process_input(user_input, client_id)
        st.session_state.chat_history.append({"user": user_input, "digiman": response})
        st.write(f"Digiman: {response}")

    st.subheader("Conversation History")
    for chat in st.session_state.chat_history[-5:]:
        st.write(f"**You**: {chat['user']}")
        st.write(f"**Digiman**: {chat['digiman']}")

    missing_keys = audit_env_keys(["XAI_API_KEY", "HUBSPOT_API_KEY", "WEBFLOW_CLIENT_ID", "WEBFLOW_CLIENT_SECRET"])
    if missing_keys:
        st.warning(f"Missing API keys: {', '.join(missing_keys)}. Some features may be disabled.")

# Main Loop
def main_loop_forever():
    flask_thread = threading.Thread(target=app.run, kwargs={"host": "0.0.0.0", "port": 5001})
    flask_thread.daemon = True
    flask_thread.start()
    
    cli_thread = threading.Thread(target=cli_loop)
    cli_thread.daemon = True
    cli_thread.start()
    
    time.sleep(1)
    while True:
        try:
            autonomous_loop()
        except Exception as e:
            logger.error(f"Autonomous loop error: {e}")
        time.sleep(10)

if __name__ == "__main__":
    print("DigiMan 6.0.3 – Autonomous Business OS with Franchise Intelligence")
    if os.getenv("STREAMLIT_RUN"):
        streamlit_app()
    else:
        main_loop_forever()