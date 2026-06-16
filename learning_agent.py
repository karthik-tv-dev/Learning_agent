import json
import os
import re
import string

from datetime import datetime
from typing import Optional


class LearningAgent:
    

    def __init__(self, knowledge_file: str = "agent_knowledge.json"):
        
        self.knowledge_file = knowledge_file
        self.knowledge_base = []
        self.conversation_history = []
        self.learning_mode = False
        self.pending_input = None
        self.interaction_count = 0
        self.load_knowledge()

    def _normalize_text(self, text: str) -> str:
        
        text = text.lower().strip()
        text = text.translate(str.maketrans("", "", string.punctuation))
        text = re.sub(r"\s+", " ", text)
        return text

    def _tokenize(self, text: str) -> list:
        
        return self._normalize_text(text).split()

    def _similarity_score(self, text1: str, text2: str) -> float:
        
        tokens1 = set(self._tokenize(text1))
        tokens2 = set(self._tokenize(text2))

        if not tokens1 or not tokens2:
            return 0.0

        intersection = tokens1 & tokens2
        union = tokens1 | tokens2

        return len(intersection) / len(union)

    def load_knowledge(self) -> None:
       
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
        
        data = {
            "knowledge": self.knowledge_base,
            "interaction_count": self.interaction_count,
            "last_updated": datetime.now().isoformat(),
        }
        with open(self.knowledge_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _find_best_response(self, user_input: str) -> tuple[Optional[str], float]:
        
        if not self.knowledge_base:
            return None, 0.0

        best_match = None
        best_score = 0.0
        normalized_input = self._normalize_text(user_input)

        for entry in self.knowledge_base:
            score = self._similarity_score(normalized_input, entry["input"])
            
            weighted_score = score * min(entry.get("confidence", 1), 10) / 10

            if weighted_score > best_score:
                best_score = weighted_score
                best_match = entry

        if best_match and best_score >= 0.3: 
            return best_match["response"], best_score

        return None, 0.0

    def _learn_response(self, user_input: str, agent_response: str) -> None:
        
        normalized_input = self._normalize_text(user_input)

        
        for entry in self.knowledge_base:
            if self._similarity_score(normalized_input, entry["input"]) > 0.8:
                # Reinforce existing entry
                entry["confidence"] = entry.get("confidence", 1) + 1
                entry["response"] = agent_response  
                print(f"[Agent] Reinforced existing knowledge (confidence: {entry['confidence']})")
                self.save_knowledge()
                return

        
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
        
        
        defaults = [
            "I'm not sure I understand. Could you rephrase that?",
            "I haven't learned about that yet. Can you help me understand?",
            "Interesting! I don't have a response for that yet. Want to teach me?",
            "I'm still learning. Could you explain what you mean?",
        ]
        
        return defaults[self.interaction_count % len(defaults)]

    def respond(self, user_input: str) -> str:
        
        
        self.interaction_count += 1

        
        if self.learning_mode:
            return self._handle_learning_mode(user_input)

        
        normalized = self._normalize_text(user_input)

        if normalized in ["learn mode", "teach me", "enable learning"]:
            self.learning_mode = True
            return 

        if normalized in ["exit learn mode", "disable learning", "stop learning"]:
            self.learning_mode = False
            return 

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
                return 
            return 

        
        response, confidence = self._find_best_response(user_input)

        if response:
            self.conversation_history.append({
                "timestamp": datetime.now().isoformat(),
                "user": user_input,
                "agent": response,
                "confidence": confidence,
            })
            return f"{response} (confidence: {confidence:.0%})"

        
        self.pending_input = user_input
        self.learning_mode = True
        default = self._get_default_response()
        return f"{default}\n\nI'm in learning mode now. Please teach me how to respond to: \"{user_input}\""

    def _handle_learning_mode(self, user_input: str) -> str:
        
        if self.pending_input:
            
            self._learn_response(self.pending_input, user_input)
            self.pending_input = None
            self.learning_mode = False
            return 
        else:
            
            response, confidence = self._find_best_response(user_input)
            if response:
                self.learning_mode = False
                return f"{response} (confidence: {confidence:.0%})"
            else:
                self.pending_input = user_input
                return f"I don't have a response for that yet. Please teach me how to respond to: \"{user_input}\""

    def get_conversation_history(self) -> list:
        
        return self.conversation_history

    def export_knowledge(self, filename: str = None) -> str:
        
        if filename is None:
            filename = f"knowledge_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

        with open(filename, "w", encoding="utf-8") as f:
            f.write("Learning Agent Knowledge Base\n")
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
                    for entry in history[-10:]:  
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
    
    
    
    
    
    
    
    
    