#!/usr/bin/env python3
"""
Process newly downloaded datasets into high-quality conversational format for agentic TinyLlama training.
Converts CVE, Breaches, and NER datasets into multi-turn conversations.
"""

import json
import pandas as pd
import random
from datasets import load_from_disk
from pathlib import Path
from typing import Dict, List, Any, Tuple

class NewDatasetProcessor:
    """Convert new datasets to conversational format for agentic training"""
    
    def __init__(self, base_dir: str = "/Users/danielrodrigo/Workspace/PyScience/datasets"):
        self.base_dir = Path(base_dir)
        self.output_dir = self.base_dir / "processed_new"
        self.output_dir.mkdir(exist_ok=True)
        
        # Conversation templates for agentic behavior
        self.cve_templates = [
            "Can you provide details about {cve_id}?",
            "What is {cve_id} and how severe is it?",
            "I need information on {cve_id} for my security assessment.",
            "Explain the vulnerability {cve_id} and its implications.",
            "What are the technical details of {cve_id}?",
        ]
        
        self.breach_templates = [
            "Tell me about the {entity} data breach.",
            "What happened in the {entity} security incident?",
            "Can you analyze the {entity} breach case?",
            "I need details on the data breach at {entity}.",
            "What can we learn from the {entity} security breach?",
        ]
        
        self.ner_templates = [
            "Can you identify cybersecurity entities in this text?",
            "Extract security-related terms from this content.",
            "Analyze this text for cybersecurity terminology.",
            "What cybersecurity entities are mentioned here?",
            "Help me identify threats and security terms in this text.",
        ]
    
    def process_cve_dataset(self) -> List[Dict]:
        """Convert CVE dataset to conversational format"""
        print("🔄 Processing CVE dataset...")
        
        cve_dataset = load_from_disk(str(self.base_dir / "cybersecurity_cve_dataset"))
        train_data = cve_dataset['train']
        
        conversations = []
        
        for i, sample in enumerate(train_data):
            # Extract CVE ID from instruction or outputs
            cve_text = sample['outputs']
            cve_id = "unknown CVE"
            if "CVE:" in cve_text:
                cve_line = cve_text.split('\n')[0]
                cve_id = cve_line.replace("CVE:", "").strip()
            
            # Create user question
            user_template = random.choice(self.cve_templates)
            user_question = user_template.format(cve_id=cve_id)
            
            # Create assistant response with enhanced formatting
            assistant_response = self.format_cve_response(sample['outputs'])
            
            conversation = {
                "messages": [
                    {
                        "role": "user",
                        "content": user_question
                    },
                    {
                        "role": "assistant", 
                        "content": assistant_response
                    }
                ]
            }
            
            conversations.append(conversation)
            
            if (i + 1) % 10000 == 0:
                print(f"  Processed {i+1:,}/{len(train_data):,} CVE samples...")
        
        print(f"✅ CVE processing complete: {len(conversations):,} conversations")
        return conversations
    
    def format_cve_response(self, cve_output: str) -> str:
        """Format CVE information into a helpful assistant response"""
        lines = cve_output.strip().split('\n')
        
        response_parts = []
        
        for line in lines:
            if line.startswith("CVE:"):
                cve_id = line.replace("CVE:", "").strip()
                response_parts.append(f"Here's the detailed information about {cve_id}:")
            elif line.startswith("Description:"):
                desc = line.replace("Description:", "").strip()
                response_parts.append(f"\n**Description:**\n{desc}")
            elif line.startswith("published:"):
                pub_date = line.replace("published:", "").strip()
                response_parts.append(f"\n**Published:** {pub_date}")
            elif line.strip():
                response_parts.append(f"\n{line}")
        
        if not response_parts:
            return f"I can provide this CVE information:\n\n{cve_output}"
        
        return "".join(response_parts)
    
    def process_breaches_dataset(self) -> List[Dict]:
        """Convert security breaches dataset to conversational format"""
        print("🔄 Processing Security Breaches dataset...")
        
        breaches_dataset = load_from_disk(str(self.base_dir / "cyber_security_breaches"))
        train_data = breaches_dataset['train']
        
        conversations = []
        
        for i, sample in enumerate(train_data):
            entity = sample['Name_of_Covered_Entity']
            if not entity or pd.isna(entity):
                continue
                
            # Create user question
            user_template = random.choice(self.breach_templates)
            user_question = user_template.format(entity=entity)
            
            # Create assistant response
            assistant_response = self.format_breach_response(sample)
            
            # Add follow-up for more agentic behavior
            if random.random() < 0.3:  # 30% chance of multi-turn
                conversation = {
                    "messages": [
                        {
                            "role": "user",
                            "content": user_question
                        },
                        {
                            "role": "assistant",
                            "content": assistant_response
                        },
                        {
                            "role": "user", 
                            "content": "What security measures could have prevented this breach?"
                        },
                        {
                            "role": "assistant",
                            "content": self.generate_prevention_advice(sample)
                        }
                    ]
                }
            else:
                conversation = {
                    "messages": [
                        {
                            "role": "user",
                            "content": user_question
                        },
                        {
                            "role": "assistant",
                            "content": assistant_response
                        }
                    ]
                }
            
            conversations.append(conversation)
            
            if (i + 1) % 200 == 0:
                print(f"  Processed {i+1:,}/{len(train_data):,} breach samples...")
        
        print(f"✅ Breaches processing complete: {len(conversations):,} conversations")
        return conversations
    
    def format_breach_response(self, sample: Dict) -> str:
        """Format breach information into a helpful assistant response"""
        entity = sample.get('Name_of_Covered_Entity', 'Unknown Entity')
        state = sample.get('State', 'Unknown')
        affected = sample.get('Individuals_Affected', 'Unknown number of')
        breach_type = sample.get('Type_of_Breach', 'Unknown type')
        location = sample.get('Location_of_Breached_Information', 'Unknown location')
        date = sample.get('Date_of_Breach', 'Unknown date')
        summary = sample.get('Summary', '')
        
        response = f"Here are the details about the {entity} data breach:\n\n"
        response += f"**Organization:** {entity} ({state})\n"
        response += f"**Individuals Affected:** {affected:,} people\n" if isinstance(affected, (int, float)) else f"**Individuals Affected:** {affected}\n"
        response += f"**Breach Type:** {breach_type}\n"
        response += f"**Information Location:** {location}\n"
        response += f"**Date of Breach:** {date}\n"
        
        if summary and not pd.isna(summary):
            response += f"\n**Summary:**\n{summary}"
        
        return response
    
    def generate_prevention_advice(self, sample: Dict) -> str:
        """Generate security advice based on breach type"""
        breach_type = sample.get('Type_of_Breach', '').lower()
        location = sample.get('Location_of_Breached_Information', '').lower()
        
        advice = "Based on this breach, here are key prevention measures:\n\n"
        
        if 'theft' in breach_type:
            advice += "• **Physical Security:** Implement secure storage and access controls\n"
            advice += "• **Data Encryption:** Encrypt sensitive data both at rest and in transit\n"
            advice += "• **Access Management:** Limit who can access sensitive information\n"
        elif 'hack' in breach_type or 'cyber' in breach_type:
            advice += "• **Network Security:** Deploy firewalls and intrusion detection systems\n"
            advice += "• **Patch Management:** Keep all systems updated with security patches\n"
            advice += "• **Multi-Factor Authentication:** Implement strong authentication controls\n"
        elif 'unauthorized' in breach_type:
            advice += "• **Access Controls:** Implement role-based access management\n"
            advice += "• **Monitoring:** Deploy user activity monitoring systems\n"
            advice += "• **Training:** Conduct regular security awareness training\n"
        
        if 'paper' in location:
            advice += "• **Document Security:** Secure physical document storage and disposal\n"
        elif 'computer' in location or 'electronic' in location:
            advice += "• **Data Protection:** Implement data loss prevention solutions\n"
        
        advice += "• **Incident Response:** Develop and test incident response procedures\n"
        advice += "• **Regular Audits:** Conduct periodic security assessments"
        
        return advice
    
    def process_ner_dataset(self) -> List[Dict]:
        """Convert NER dataset to conversational format"""
        print("🔄 Processing Cybersecurity NER dataset...")
        
        ner_dataset = load_from_disk(str(self.base_dir / "cybersecurity_ner"))
        train_data = ner_dataset['train']
        
        conversations = []
        
        # NER tag mapping (based on common cybersecurity NER schemes)
        tag_names = {
            0: 'O',      # Outside
            1: 'B-MAL',  # Begin Malware
            2: 'I-MAL',  # Inside Malware
            3: 'B-TOOL', # Begin Tool
            4: 'I-TOOL', # Inside Tool
            5: 'B-ATT',  # Begin Attack
            6: 'I-ATT',  # Inside Attack
            7: 'B-VUL',  # Begin Vulnerability
            8: 'I-VUL',  # Inside Vulnerability
            9: 'B-LOC',  # Begin Location
            10: 'I-LOC', # Inside Location
        }
        
        for i, sample in enumerate(train_data):
            tokens = sample['tokens']
            ner_tags = sample['ner_tags']
            
            # Reconstruct text
            text = " ".join(tokens)
            
            # Extract entities
            entities = self.extract_entities(tokens, ner_tags, tag_names)
            
            # Create user question
            user_template = random.choice(self.ner_templates)
            user_question = f"{user_template}\n\nText: \"{text}\""
            
            # Create assistant response
            assistant_response = self.format_ner_response(text, entities)
            
            conversation = {
                "messages": [
                    {
                        "role": "user",
                        "content": user_question
                    },
                    {
                        "role": "assistant",
                        "content": assistant_response
                    }
                ]
            }
            
            conversations.append(conversation)
            
            if (i + 1) % 500 == 0:
                print(f"  Processed {i+1:,}/{len(train_data):,} NER samples...")
        
        print(f"✅ NER processing complete: {len(conversations):,} conversations")
        return conversations
    
    def extract_entities(self, tokens: List[str], tags: List[int], tag_names: Dict) -> List[Tuple[str, str]]:
        """Extract named entities from tokens and tags"""
        entities = []
        current_entity = []
        current_type = None
        
        for token, tag in zip(tokens, tags):
            tag_name = tag_names.get(tag, 'O')
            
            if tag_name.startswith('B-'):
                # Save previous entity if exists
                if current_entity and current_type:
                    entities.append((" ".join(current_entity), current_type))
                
                # Start new entity
                current_entity = [token]
                current_type = tag_name[2:]  # Remove 'B-'
                
            elif tag_name.startswith('I-') and current_type == tag_name[2:]:
                # Continue current entity
                current_entity.append(token)
                
            else:
                # Save entity and reset
                if current_entity and current_type:
                    entities.append((" ".join(current_entity), current_type))
                current_entity = []
                current_type = None
        
        # Save final entity
        if current_entity and current_type:
            entities.append((" ".join(current_entity), current_type))
        
        return entities
    
    def format_ner_response(self, text: str, entities: List[Tuple[str, str]]) -> str:
        """Format NER results into helpful assistant response"""
        if not entities:
            return f"I analyzed the text but didn't find any clearly identified cybersecurity entities. The text appears to be: \"{text}\""
        
        response = "I've identified the following cybersecurity entities in the text:\n\n"
        
        entity_types = {}
        for entity, entity_type in entities:
            if entity_type not in entity_types:
                entity_types[entity_type] = []
            entity_types[entity_type].append(entity)
        
        type_descriptions = {
            'MAL': 'Malware',
            'TOOL': 'Security Tools',
            'ATT': 'Attack Methods',
            'VUL': 'Vulnerabilities',
            'LOC': 'Locations/Systems'
        }
        
        for entity_type, entity_list in entity_types.items():
            type_name = type_descriptions.get(entity_type, entity_type)
            response += f"**{type_name}:**\n"
            for entity in set(entity_list):  # Remove duplicates
                response += f"• {entity}\n"
            response += "\n"
        
        return response.strip()
    
    def merge_all_datasets(self) -> Tuple[str, str]:
        """Process all datasets and merge into final training files"""
        print("🚀 Starting comprehensive dataset processing...")
        
        all_conversations = []
        
        # Process each dataset
        cve_conversations = self.process_cve_dataset()
        all_conversations.extend(cve_conversations)
        
        breach_conversations = self.process_breaches_dataset()
        all_conversations.extend(breach_conversations)
        
        ner_conversations = self.process_ner_dataset()
        all_conversations.extend(ner_conversations)
        
        print(f"\n📊 Total conversations: {len(all_conversations):,}")
        
        # Shuffle for better training
        random.shuffle(all_conversations)
        
        # Create train/validation split (90/10)
        split_idx = int(len(all_conversations) * 0.9)
        train_conversations = all_conversations[:split_idx]
        val_conversations = all_conversations[split_idx:]
        
        # Save to jsonl files
        train_path = self.output_dir / "enhanced_cybersec_training.jsonl"
        val_path = self.output_dir / "enhanced_cybersec_validation.jsonl"
        
        with open(train_path, 'w') as f:
            for conv in train_conversations:
                f.write(json.dumps(conv) + '\n')
        
        with open(val_path, 'w') as f:
            for conv in val_conversations:
                f.write(json.dumps(conv) + '\n')
        
        print(f"✅ Train data: {train_path} ({len(train_conversations):,} conversations)")
        print(f"✅ Validation data: {val_path} ({len(val_conversations):,} conversations)")
        
        return str(train_path), str(val_path)

def main():
    """Main processing function"""
    
    processor = NewDatasetProcessor()
    train_path, val_path = processor.merge_all_datasets()
    
    print(f"\n🎉 Dataset processing complete!")
    print(f"📁 Training data: {train_path}")
    print(f"📁 Validation data: {val_path}")
    print(f"\n🔗 Ready for MLX training with existing training_config.yaml (alpha=128)")

if __name__ == "__main__":
    main()
