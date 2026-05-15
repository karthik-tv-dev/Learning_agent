"""
Learning AI Agent - An agent that improves over time based on interactions.

This agent learns from conversations by:
1. Storing conversation patterns in a knowledge base
2. Finding similar patterns to respond appropriately
3. Learning new responses when it doesn't know how to reply
4. Using text similarity matching to find relevant responses
"""

import json
import os
import re
import string
from collections import Counter
from datetime import datetime
from typing import Optional


class LearningAgent:
    """A learning AI agent that improves through interactions."""

    def __init__(self, knowledge_file: str = "agent_knowledge.json"):
        """
        Initialize the learning agent.

        Args:
            knowledge_file: Path to the file where knowledge is stored
        """
        self.knowledge_file = knowledge_file
        self.knowledge_base = []
        self.conversation_history = []
        self.learning_mode = False
        self.pending_input = None
        self.interaction_count = 0
        self.load_knowledge()

    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison by lowercasing and removing punctuation."""
        text = text.lower().strip()
        text = text.translate(str.maketrans("", "", string.punctuation))
        text = re.sub(r"\s+", " ", text)
        return text

    def _tokenize(self, text: str) -> list:
        """Tokenize text into words."""
        return self._normalize_text(text).split()

    def _similarity_score(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two texts using word overlap.

        Returns:
            A score between 0 and 1 indicating similarity
        """
        tokens1 = set(self._tokenize(text1))
        tokens2 = set(self._tokenize(text2))

        if not tokens1 or not tokens2:
            return 0.0

        intersection = tokens1 & tokens2
        union = tokens1 | tokens2

        return len(intersection) / len(union)

    def load_knowledge(self) -> None:
        """Load the knowledge base from file."""
        if os.path.exists(self.knowledge_file):
            try:
                with open(self.knowledge_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.knowledge_base = data.get("knowledge", [])
                    self.interaction_count = data.get("interaction_count", 0)
                print(f"[Agent] Loaded {len(self.knowledge_base)} knowledge entries.")
            except (json.JSONDecodeError, IOError) as e:
                print(f"[Agent] Error loading knowledge: {e}")
                self.knowledge_base = []
        else:
            print("[Agent] No existing knowledge base found. Starting fresh.")
            # Add some initial greetings
            self._add_initial_knowledge()

    def _add_initial_knowledge(self) -> None:
        """Add some initial knowledge to help the agent get started."""
        initial_knowledge = [
            {"input": "hello", "response": "Hello! How can I assist you today?", "confidence": 3},
            {"input": "hi", "response": "Hi there! What can I do for you?", "confidence": 3},
            {"input": "how are you", "response": "I'm doing well, thank you for asking! How about you?", "confidence": 3},
            {"input": "what is your name", "response": "I'm a Learning Agent, designed to improve through our conversations.", "confidence": 3},
            {"input": "who are you", "response": "I'm an AI agent that learns from our interactions. The more we talk, the smarter I get!", "confidence": 3},
            {"input": "bye", "response": "Goodbye! Have a great day!", "confidence": 3},
            {"input": "thank you", "response": "You're welcome! Is there anything else I can help with?", "confidence": 3},
            {"input": "what can you do", "response": "I can converse with you and learn from our interactions. I store what I learn and get better at responding over time!", "confidence": 3},
            {"input": "help", "response": "I'm here to help! Just type your message and I'll do my best to respond. If I don't know something, you can teach me!", "confidence": 3},
        ]
        self.knowledge_base.extend(initial_knowledge)
        self.save_knowledge()
        print(f"[Agent] Added {len(initial_knowledge)} initial knowledge entries.")

    def save_knowledge(self) -> None:
        """Save the knowledge base to file."""
        data = {
            "knowledge": self.knowledge_base,
            "interaction_count": self.interaction_count,
            "last_updated": datetime.now().isoformat(),
        }
        with open(self.knowledge_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _find_best_response(self, user_input: str) -> tuple[Optional[str], float]:
        """
        Find the best matching response for the given input.

        Returns:
            A tuple of (response, confidence_score)
        """
        if not self.knowledge_base:
            return None, 0.0

        best_match = None
        best_score = 0.0
        normalized_input = self._normalize_text(user_input)

        for entry in self.knowledge_base:
            score = self._similarity_score(normalized_input, entry["input"])
            # Weight by confidence (how many times this has been reinforced)
            weighted_score = score * min(entry.get("confidence", 1), 10) / 10

            if weighted_score > best_score:
                best_score = weighted_score
                best_match = entry

        if best_match and best_score >= 0.3:  # Minimum threshold
            return best_match["response"], best_score

        return None, 0.0

    def _learn_response(self, user_input: str, agent_response: str) -> None:
        """
        Learn a new input-response pair.

        Args:
            user_input: The user's input text
            agent_response: The response the agent should learn
        """
        normalized_input = self._normalize_text(user_input)

        # Check if similar input already exists
        for entry in self.knowledge_base:
            if self._similarity_score(normalized_input, entry["input"]) > 0.8:
                # Reinforce existing entry
                entry["confidence"] = entry.get("confidence", 1) + 1
                entry["response"] = agent_response  # Update with new response
                print(f"[Agent] Reinforced existing knowledge (confidence: {entry['confidence']})")
                self.save_knowledge()
                return

        # Add new knowledge entry
        new_entry = {
            "input": normalized_input,
            "response": agent_response,
            "confidence": 1,
            "learned_at": datetime.now().isoformat(),
        }
        self.knowledge_base.append(new_entry)
        self.save_knowledge()
        print(f"[Agent] Learned new knowledge! Total entries: {len(self.knowledge_base)}")

    def _get_default_response(self) -> str:
        """Get a default response when the agent doesn't understand."""
        defaults = [
            "I'm not sure I understand. Could you rephrase that?",
            "I haven't learned about that yet. Can you help me understand?",
            "Interesting! I don't have a response for that yet. Want to teach me?",
            "I'm still learning. Could you explain what you mean?",
        ]
        # Use interaction count to vary responses
        return defaults[self.interaction_count % len(defaults)]

    def respond(self, user_input: str) -> str:
        """
        Generate a response to the user's input.

        Args:
            user_input: The user's message

        Returns:
            The agent's response
        """
        self.interaction_count += 1

        # Handle learning mode
        if self.learning_mode:
            return self._handle_learning_mode(user_input)

        # Handle special commands
        normalized = self._normalize_text(user_input)

        if normalized in ["learn mode", "teach me", "enable learning"]:
            self.learning_mode = True
            return "Learning mode enabled! Send me an input and I'll ask you to teach me the response."

        if normalized in ["exit learn mode", "disable learning", "stop learning"]:
            self.learning_mode = False
            return "Learning mode disabled."

        if normalized in ["stats", "status", "how smart are you"]:
            return (
                f"I have {len(self.knowledge_base)} knowledge entries and have had "
                f"{self.interaction_count} interactions. My knowledge base grows with every conversation!"
            )

        if normalized in ["clear memory", "reset knowledge"]:
            confirm = input("Are you sure you want to clear all learned knowledge? (yes/no): ")
            if confirm.lower() in ["yes", "y"]:
                self.knowledge_base = []
                self._add_initial_knowledge()
                return "Memory cleared! I'm back to my initial knowledge."
            return "Memory clear cancelled."

        # Find and return the best response
        response, confidence = self._find_best_response(user_input)

        if response:
            self.conversation_history.append({
                "timestamp": datetime.now().isoformat(),
                "user": user_input,
                "agent": response,
                "confidence": confidence,
            })
            return f"{response} (confidence: {confidence:.0%})"

        # If no good match, offer to learn
        self.pending_input = user_input
        self.learning_mode = True
        default = self._get_default_response()
        return f"{default}\n\nI'm in learning mode now. Please teach me how to respond to: \"{user_input}\""

    def _handle_learning_mode(self, user_input: str) -> str:
        """Handle input when in learning mode."""
        if self.pending_input:
            # Learn the response for the pending input
            self._learn_response(self.pending_input, user_input)
            self.pending_input = None
            self.learning_mode = False
            return "Got it! I've learned that response. I'm now back to normal mode."
        else:
            # User sent input without a pending question
            # Try to respond normally first
            response, confidence = self._find_best_response(user_input)
            if response:
                self.learning_mode = False
                return f"{response} (confidence: {confidence:.0%})"
            else:
                self.pending_input = user_input
                return f"I don't have a response for that yet. Please teach me how to respond to: \"{user_input}\""

    def get_conversation_history(self) -> list:
        """Get the conversation history."""
        return self.conversation_history

    def export_knowledge(self, filename: str = None) -> str:
        """Export the knowledge base as readable text."""
        if filename is None:
            filename = f"knowledge_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

        with open(filename, "w", encoding="utf-8") as f:
            f.write(f"Learning Agent Knowledge Base\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n")
            f.write(f"Total entries: {len(self.knowledge_base)}\n")
            f.write(f"Total interactions: {self.interaction_count}\n")
            f.write("=" * 50 + "\n\n")

            for i, entry in enumerate(self.knowledge_base, 1):
                f.write(f"Entry {i}:\n")
                f.write(f"  Input: {entry['input']}\n")
                f.write(f"  Response: {entry['response']}\n")
                f.write(f"  Confidence: {entry.get('confidence', 1)}\n")
                if 'learned_at' in entry:
                    f.write(f"  Learned: {entry['learned_at']}\n")
                f.write("\n")

        return filename


def main():
    """Main function to run the learning agent in interactive mode."""
    print("=" * 60)
    print("  LEARNING AI AGENT")
    print("  An agent that improves through interactions!")
    print("=" * 60)
    print()

    agent = LearningAgent()

    print("\nCommands:")
    print("  - 'learn mode' : Enable learning mode")
    print("  - 'disable learning' : Disable learning mode")
    print("  - 'stats' : Show agent statistics")
    print("  - 'export' : Export knowledge base to file")
    print("  - 'history' : Show conversation history")
    print("  - 'quit' or 'exit' : End the conversation")
    print()

    while True:
        try:
            user_input = input("You: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ["quit", "exit", "bye"]:
                print("Agent: Goodbye! I've saved everything I learned.")
                break

            if user_input.lower() == "export":
                filename = agent.export_knowledge()
                print(f"Agent: Knowledge exported to {filename}")
                continue

            if user_input.lower() == "history":
                history = agent.get_conversation_history()
                if history:
                    print("\n--- Conversation History ---")
                    for entry in history[-10:]:  # Show last 10 entries
                        print(f"  You: {entry['user']}")
                        print(f"  Agent: {entry['agent']}")
                        print()
                else:
                    print("Agent: No conversation history yet.")
                continue

            response = agent.respond(user_input)
            print(f"Agent: {response}")
            print()

        except KeyboardInterrupt:
            print("\n\nAgent: Interrupted! Goodbye!")
            break
        except EOFError:
            print("\n\nAgent: End of input! Goodbye!")
            break


if __name__ == "__main__":
    main()