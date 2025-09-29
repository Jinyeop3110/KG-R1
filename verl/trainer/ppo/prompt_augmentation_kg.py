# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Prompt augmentation utilities for Knowledge Graph-based question answering.

This module provides instruction guidelines and augmentation functionality
to help models learn proper KG query formatting and reasoning patterns.

Available guideline levels:
- "extensive": Full detailed instructions with comprehensive explanations
- "detailed": Default guidelines with 5 query limit + KG prioritization
- "detailed_hierarchical": Hierarchical format examples with query functions
- "detailed_flat": Flat format examples with query functions
- "detailed_flat_turn7": Multi-turn reasoning with up to 7 turns and 7 kg-queries
- "detailed_flat_multiTQ": MultiTQ-specific temporal reasoning guidelines with timestamp filtering
- "detailed_flat_multiTQ_v2": Enhanced MultiTQ variant with improved temporal analysis and entity extraction strategies (7 queries)
- "detailed_flat_multiTQ_v2_q5": Enhanced MultiTQ variant with improved temporal analysis and entity extraction strategies (5 queries)
- "detailed_flat_multiTQ_v3": Advanced MultiTQ variant with multi-phase temporal reasoning and comprehensive analytical framework
- "detailed_flat_10arxiv": 10ArXiv-specific research paper domain reasoning with academic terminology
- "detailed_minimal": Minimal prompt with basic KG functions only (for clean evaluation without extensive guidance)
- "detailed_minimal_5q": General minimal prompt with 5 kg-queries limit (clean evaluation)
- "vanilla": Simple direct answering without KG or reasoning instructions
- "cot": Chain-of-Thought prompting with reasoning steps before answers
"""

import re
from typing import Optional


VANILLA_GUIDELINE = """
Answer the given question directly and concisely based on your knowledge.
Do not use any special formatting, reasoning tags, or external tools.
Provide a clear, factual answer.
"""

COT_GUIDELINE = """
Think through the question step by step, then provide the answer.
Show your reasoning process before giving the final answer.
Do not use any external tools or knowledge graphs.
"""


def extract_clean_question_from_prompt(prompt_content: str) -> str:
    """
    Extract clean question from KG-augmented prompt content for vanilla evaluation.
    
    This function parses prompts that contain KG instructions and extracts just
    the raw question text, removing initial entity hints and KG formatting.
    
    Args:
        prompt_content: Full prompt content with potential KG instructions
        
    Returns:
        Clean question text without KG instructions or initial entity hints
    """
    # Find the question part - look for "Question:" marker
    if 'Question:' in prompt_content:
        question_part = prompt_content.split('Question:')[1].strip()
        
        # Remove initial entities hint: "question text?? (Initial entities: "entity1", "entity2")"
        if '(Initial entities:' in question_part:
            clean_question = question_part.split('(Initial entities:')[0].strip()
        else:
            clean_question = question_part
            
        return clean_question
    else:
        # Fallback: if no "Question:" found, try to clean the entire content
        # Remove common KG instruction patterns
        content = prompt_content
        
        # Remove the main KG instruction block
        if 'You must conduct reasoning inside <think>' in content:
            # Find where the instruction block ends - usually before the actual question
            content_lines = content.split('\n')
            question_lines = []
            found_question = False
            
            for line in content_lines:
                # Skip instruction lines
                if any(pattern in line for pattern in [
                    'You must conduct reasoning inside <think>',
                    'you can query the knowledge graph',
                    '<kg-query>',
                    '<information>',
                    'Beijing </answer>'
                ]):
                    continue
                
                # Look for the actual question
                if line.strip() and not found_question:
                    # This might be the question
                    if '?' in line or line.endswith('.'):
                        question_lines.append(line.strip())
                        found_question = True
                elif found_question:
                    # Continue capturing multi-line questions
                    if line.strip():
                        question_lines.append(line.strip())
                    else:
                        break
            
            if question_lines:
                clean_question = ' '.join(question_lines)
                # Remove initial entities hint if present
                if '(Initial entities:' in clean_question:
                    clean_question = clean_question.split('(Initial entities:')[0].strip()
                return clean_question
        
        # Final fallback: return the content as-is but remove initial entities
        if '(Initial entities:' in content:
            return content.split('(Initial entities:')[0].strip()
            
        return content.strip()


def create_vanilla_prompt(question: str) -> str:
    """
    Create a clean vanilla prompt for direct question answering.
    
    Args:
        question: Clean question text
        
    Returns:
        Simple vanilla prompt with direct instructions and examples
    """
    return f"""You are Qwen, created by Alibaba Cloud. You are a helpful assistant.

Answer the given question directly and concisely based on your knowledge.
Do not use any special formatting, reasoning tags, or external tools.
Format your answer as: Answers: [ "answer1", "answer2", ... ]
For single answers, use: Answers: [ "answer" ]
Provide a clear, factual answer.

Question: {question}
Answers:"""


def create_cot_prompt(question: str) -> str:
    """
    Create a Chain-of-Thought (COT) prompt that encourages reasoning before answering.
    
    Args:
        question: Clean question text
        
    Returns:
        COT prompt with reasoning instructions and examples
    """
    return f"""You are Qwen, created by Alibaba Cloud. You are a helpful assistant.

Think through the question step by step, then provide the answer.

IMPORTANT: Follow this exact format:
1. Start with "Reasoning:" followed by your step-by-step thinking
2. End with "Answers:" followed by your final answer in brackets
3. Do NOT put "Answers:" before your reasoning

Format: 
Reasoning: [your step-by-step thinking process]
Answers: [ "answer1", "answer2", ... ]

For single answers, use: Answers: [ "answer" ]

Question: {question}
Reasoning:"""



DETAILED_GUIDELINE = """
You are allowed to make up to 7 kg-queries. 
If you encounter a KG-related error, read the error message carefully and correct your query.

Use exactly these query functions:
- get_tail_relations(entity): Returns relations where the entity is the subject/head.
- get_head_relations(entity): Returns relations where the entity is the object/tail.
- get_tail_entities(entity, relation): Returns entities connected to the given entity by the specified relation.
- get_head_entities(entity, relation): Returns entities from which the given entity is connected by the specified relation.

IMPORTANT:
- Always begin with think after getting question or information.
- Always prefer information retrieved from the KG over your internal knowledge.
- Use KG data as your primary source if relevant information is available.

Examples of entities:
- Named entities: "Barack Obama", "Taylor Swift", "Albert Einstein", "New York City", "France", "Mount Everest", "Google", "United Nations", "Harvard University"
- Entity IDs: "m.02mjmr", "m.09c7w0"

Examples of relations:
- "people.person.nationality"
- "people.person.spouse_s"
- "location.location.contains"
- "location.country.capital"
- "location.location.nearby_airports"
- "organization.organization.headquarters"
- "organization.organization.founders"
- "type.object.name"
- "common.topic.notable_for"

KG Query Examples:
- get_tail_relations("Bahamas")
- get_head_relations("Barack Obama")
- get_tail_entities("Bahamas", "location.location.contains")
- get_head_entities("Barack Obama", "people.person.nationality")
"""

DETAILED_GUIDELINE_HIERARCHICAL = """
You are allowed to make up to 5 kg-queries. 
If you encounter a KG-related error, read the error message carefully and correct your query.

Use exactly these query functions:
- get_tail_relations(entity): Returns relations where the entity is the subject/head.
- get_head_relations(entity): Returns relations where the entity is the object/tail.
- get_tail_entities(entity, relation): Returns entities connected to the given entity by the specified relation.
- get_head_entities(entity, relation): Returns entities from which the given entity is connected by the specified relation.

Relations are returned in hierarchical format grouped by domain → type → property. Example for get_head_relations:
Head relations for entity "Barack Obama":
people
  person
    nationality
    spouse_s
location
  country
    capital

IMPORTANT:
- Always begin with think after getting question or information.
- Always prefer information retrieved from the KG over your internal knowledge.
- Use KG data as your primary source if relevant information is available.

Examples of entities:
- Named entities: "Barack Obama", "Taylor Swift", "Albert Einstein", "New York City", "France", "Mount Everest", "Google", "United Nations", "Harvard University"
- Entity IDs: "m.02mjmr", "m.09c7w0"

Examples of relations:
- "people.person.nationality"
- "people.person.spouse_s"
- "location.location.contains"
- "location.country.capital"
- "location.location.nearby_airports"
- "organization.organization.headquarters"
- "organization.organization.founders"
- "type.object.name"
- "common.topic.notable_for"

KG Query Examples:
- get_tail_relations("Bahamas")
- get_head_relations("Barack Obama")
- get_tail_entities("Bahamas", "location.location.contains")
- get_head_entities("Barack Obama", "people.person.nationality")
"""

DETAILED_GUIDELINE_FLAT = """
You are allowed to make up to 5 kg-queries. 
If you encounter a KG-related error, read the error message carefully and correct your query.

Use exactly these query functions:
- get_tail_relations(entity): Returns relations where the entity is the subject/head.
- get_head_relations(entity): Returns relations where the entity is the object/tail.
- get_tail_entities(entity, relation): Returns entities connected to the given entity by the specified relation.
- get_head_entities(entity, relation): Returns entities from which the given entity is connected by the specified relation.

IMPORTANT:
- Always begin with think after getting question or information.
- Always prefer information retrieved from the KG over your internal knowledge.
- Use KG data as your primary source if relevant information is available.

Examples of entities:
- Named entities: "Barack Obama", "Taylor Swift", "Albert Einstein", "New York City", "France", "Mount Everest", "Google", "United Nations", "Harvard University"
- Entity IDs: "m.02mjmr", "m.09c7w0"

Examples of relations:
- "people.person.nationality"
- "people.person.spouse_s"
- "location.location.contains"
- "location.country.capital"
- "location.location.nearby_airports"
- "organization.organization.headquarters"
- "organization.organization.founders"
- "type.object.name"
- "common.topic.notable_for"

KG Query Examples:
- get_tail_relations("Bahamas")
- get_head_relations("Barack Obama")
- get_tail_entities("Bahamas", "location.location.contains")
- get_head_entities("Barack Obama", "people.person.nationality")
"""

DETAILED_GUIDELINE_FLAT_TURN7 = """
You are allowed to make up to 7 kg-queries across multiple reasoning turns. 
If you encounter a KG-related error, read the error message carefully and correct your query.

Use exactly these query functions:
- get_tail_relations(entity): Returns relations where the entity is the subject/head.
- get_head_relations(entity): Returns relations where the entity is the object/tail.
- get_tail_entities(entity, relation): Returns entities connected to the given entity by the specified relation.
- get_head_entities(entity, relation): Returns entities from which the given entity is connected by the specified relation.

IMPORTANT:
- Always begin with think after getting question or information.
- You can perform multi-turn reasoning across up to 7 turns to solve complex questions.
- Always prefer information retrieved from the KG over your internal knowledge.
- Use KG data as your primary source if relevant information is available.
- Build upon information from previous turns to answer complex multi-step questions.

Examples of entities:
- Named entities: "Barack Obama", "Taylor Swift", "Albert Einstein", "New York City", "France", "Mount Everest", "Google", "United Nations", "Harvard University"
- Entity IDs: "m.02mjmr", "m.09c7w0"

Examples of relations:
- "people.person.nationality"
- "people.person.spouse_s"
- "location.location.contains"
- "location.country.capital"
- "location.location.nearby_airports"
- "organization.organization.headquarters"
- "organization.organization.founders"
- "type.object.name"
- "common.topic.notable_for"

KG Query Examples:
- get_tail_relations("Bahamas")
- get_head_relations("Barack Obama")
- get_tail_entities("Bahamas", "location.location.contains")
- get_head_entities("Barack Obama", "people.person.nationality")
"""

DETAILED_GUIDELINE_FLAT_MULTITQ = """
You are allowed to make up to 5 kg-queries for temporal reasoning. 
If you encounter a KG-related error, read the error message carefully and correct your query.

Use exactly these query functions for MultiTQ temporal knowledge graph:
- get_tail_relations(entity): Returns relations where the entity is the subject/head.
- get_head_relations(entity): Returns relations where the entity is the object/tail.
- get_tail_entities(entity, relation): Returns entities connected to the given entity by the specified relation, with timepoints.
- get_head_entities(entity, relation): Returns entities from which the given entity is connected by the specified relation, with timepoints.

Note: Entities returned by get_tail_entities and get_head_entities include timestamps in brackets (e.g., "Barack Obama [2005-01;2015-06]") showing when the relationship occurred.

IMPORTANT:
- Always begin with <think> after getting question or information.
- Pay special attention to temporal constraints (when, during, before, after, year, date, etc.)
- Always prefer information retrieved from the temporal KG over your internal knowledge.
- Your answer can be multiple entities (e.g., <answer>Barack Obama</answer>, <answer>Joe Biden</answer>) or multiple time points (e.g., <answer>2005-01</answer>, <answer>2015-06</answer>) or a single entity/time point

Examples of entities:
- Events: "al-Shabaab insurgency", "World War II", "2008 financial crisis"
- People: "Barack Obama", "Nelson Mandela", "Steve Jobs"  
- Organizations: "United Nations", "European Union", "Panhellenic Socialist Movement"
- Actor roles: "Protester (Egypt)", "Police (Philippines)", "Citizen (Greece)", "Social Worker (India)", "Criminal (Australia)", "Defense Attorney (Iraq)", "Men (India)", "Military (New Zealand)", "Armed Rebel (Russia)", "Business (Iran)", "Children (Philippines)"
- Government entities: "Government (Sudan)", "Government (Libya)", "Government (Syria)"

Examples of relations:
- "Make statement"
- "Participate in"
- "Lead"
- "Located in"
- "Member of"

KG Query Examples:
- get_tail_relations("Barack Obama")
- get_head_relations("al-Shabaab insurgency")
- get_tail_entities("Barack Obama", "Make statement")
- get_head_entities("United Kingdom", "Participate in")
"""

DETAILED_GUIDELINE_FLAT_MULTITQ_V2 = """
You are allowed to make up to 7 kg-queries for enhanced temporal reasoning on MultiTQ questions. 
If you encounter a KG-related error, read the error message carefully and correct your query.

Use exactly these query functions for MultiTQ temporal knowledge graph:
- get_tail_relations(entity): Returns relations where the entity is the subject/head.
- get_head_relations(entity): Returns relations where the entity is the object/tail.
- get_tail_entities(entity, relation): Returns entities connected to the given entity by the specified relation, with timepoints.
- get_head_entities(entity, relation): Returns entities from which the given entity is connected by the specified relation, with timepoints.

Examples of entities:
- Named entities: "Barack Obama", "Angela Merkel", "Xi Jinping", "Vladimir Putin", "Shinzo Abe", "Arab Spring", "Brexit referendum", "2016 U.S. election", "Ukrainian crisis", "United Nations", "European Union", "NATO", "ASEAN", "African Union", "United States", "China", "Germany", "Russia", "United Kingdom", "Government (Syria)", "Government (Libya)", "Government (Egypt)", "Protester (Egypt)", "Military (Ukraine)", "Business (China)", "Citizen (Greece)"

IMPORTANT:
- Your answer can be multiple entities (e.g., <answer>Barack Obama</answer>, <answer>Joe Biden</answer>) or multiple time points (e.g., <answer>2005-01</answer>, <answer>2015-06</answer>) or a single entity/time point
"""

DETAILED_GUIDELINE_FLAT_MULTITQ_V2_Q5 = """
You are allowed to make up to 5 kg-queries for enhanced temporal reasoning on MultiTQ questions. 
If you encounter a KG-related error, read the error message carefully and correct your query.

Use exactly these query functions for MultiTQ temporal knowledge graph:
- get_tail_relations(entity): Returns relations where the entity is the subject/head.
- get_head_relations(entity): Returns relations where the entity is the object/tail.
- get_tail_entities(entity, relation): Returns entities connected to the given entity by the specified relation, with timepoints.
- get_head_entities(entity, relation): Returns entities from which the given entity is connected by the specified relation, with timepoints.

Examples of entities:
- Named entities: "Barack Obama", "Angela Merkel", "Xi Jinping", "Vladimir Putin", "Shinzo Abe", "Arab Spring", "Brexit referendum", "2016 U.S. election", "Ukrainian crisis", "United Nations", "European Union", "NATO", "ASEAN", "African Union", "United States", "China", "Germany", "Russia", "United Kingdom", "Government (Syria)", "Government (Libya)", "Government (Egypt)", "Protester (Egypt)", "Military (Ukraine)", "Business (China)", "Citizen (Greece)"

IMPORTANT:
- Your answer can be multiple entities (e.g., <answer>Barack Obama</answer>, <answer>Joe Biden</answer>) or multiple time points (e.g., <answer>2005-01</answer>, <answer>2015-06</answer>) or a single entity/time point
"""

DETAILED_GUIDELINE_MINIMAL="""
You are allowed to make up to 7 kg-queries for research paper domain reasoning.
If you encounter a KG-related error, read the error message carefully and correct your query.

Use exactly these query functions for 10ArXiv research knowledge graph:
- get_tail_relations(entity): Returns tail relations where the entity is the subject/head (entity → tail_relation → ?).
- get_head_relations(entity): Returns head relations where the entity is the object/tail (? → head_relation → entity).
- get_tail_entities(entity, relation): Returns entities connected to the given entity by the specified relation (entity → relation → ?).
- get_head_entities(entity, relation): Returns entities from which the given entity is connected by the specified relation (? → relation → entity).

"""

DETAILED_GUIDELINE_MINIMAL_5Q = """
You are allowed to make up to 5 kg-queries.
If you encounter a KG-related error, read the error message carefully and correct your query.

Use exactly these query functions:
- get_tail_relations(entity): Returns tail relations where the entity is the subject/head.
- get_head_relations(entity): Returns head relations where the entity is the object/tail.
- get_tail_entities(entity, relation): Returns entities connected to the given entity by the specified relation.
- get_head_entities(entity, relation): Returns entities from which the given entity is connected by the specified relation.
"""

DETAILED_GUIDELINE_FLAT_10ARXIV = """
You are allowed to make up to 7 kg-queries for research paper domain reasoning.
If you encounter a KG-related error, read the error message carefully and correct your query.

Use exactly these query functions for 10ArXiv research knowledge graph:
- get_tail_relations(entity): Returns relations where the entity is the subject/head.
- get_head_relations(entity): Returns relations where the entity is the object/tail.
- get_tail_entities(entity, relation): Returns entities connected to the given entity by the specified relation.
- get_head_entities(entity, relation): Returns entities from which the given entity is connected by the specified relation.

IMPORTANT:
- Always begin with <think> after getting question or information.
- Focus on research domains, methods, frameworks, and technical relationships.
- Always prefer information retrieved from the research KG over your internal knowledge.
- Use KG data as your primary source for academic and technical information.

Examples of research entities:
- Methods: "ASTRA", "RAG", "G-RETRIEVER", "RAG 2HOP", "SELFRAG", "GRAPHRAG"
- Domains: "NATURAL LANGUAGE PROCESSING", "COMPUTER VISION", "MACHINE LEARNING", "RETRIEVAL AUGMENTED GENERATION"
- Frameworks: "TRANSFORMER", "BERT", "GPT", "LLAMA", "PYTORCH", "TENSORFLOW"
- Components: "ATTENTION MECHANISM", "ENCODER", "DECODER", "EMBEDDING", "RETRIEVER"
- Techniques: "FINE-TUNING", "PRE-TRAINING", "KNOWLEDGE DISTILLATION", "PROMPT ENGINEERING"

Examples of research relations:
- "USED_IN"
- "OUTPERFORMS"
- "EXTENDS"
- "INCLUDES"
- "ADDRESSES" 
- "IMPROVES"
- "ANALYZES"
- "HANDLES"
- "PERFORMS"
- "ENHANCES"

KG Query Examples:
- get_tail_relations("ASTRA")
- get_head_relations("RETRIEVAL AUGMENTED GENERATION")
- get_tail_entities("RAG", "USED_IN")
- get_head_entities("NATURAL LANGUAGE PROCESSING", "INCLUDES")
"""


class PromptAugmentor:
    """
    Handles prompt augmentation with KG instruction guidelines.
    
    This class provides functionality to augment prompts with instruction hints
    that help models learn proper knowledge graph querying and reasoning patterns.
    """
    
    def __init__(
        self,
        enable: bool = False,
        guideline_level: Optional[str] = None,
        hint_steps: int = 0,
        current_step: int = 0
    ):
        """
        Initialize the prompt augmentor.
        
        Args:
            enable: Whether to enable prompt augmentation
            guideline_level: Level of guidelines ("extensive", "detailed", "detailed_hierarchical", "detailed_flat", "detailed_flat_multiTQ", "detailed_flat_10arxiv", or None)
            hint_steps: Number of training steps to apply hints (0 = unlimited)
            current_step: Current training step
        """
        self.enable = enable
        self.guideline_level = guideline_level
        self.hint_steps = hint_steps
        self.current_step = current_step
        
        # Pre-compile regex patterns for performance
        self.question_patterns = [
            re.compile(r'\n\nQuestion:\s*(.+?)$', re.DOTALL),  # Question: at end with newlines
            re.compile(r'\nQuestion:\s*(.+?)$', re.DOTALL),   # Question: at end with single newline
            re.compile(r'Question:\s*(.+?)$', re.DOTALL),     # Question: at end without newlines
        ]
        
        print(f"[PROMPT_AUGMENTATION] Enabled: {self.enable}, "
              f"Level: {self.guideline_level}, Hint steps: {self.hint_steps}")
    
    def set_current_step(self, step: int):
        """Update the current training step for hint scheduling."""
        self.current_step = step
    
    def get_instruction_hint(self) -> str:
        """Get the appropriate instruction hint based on guideline level."""
        if self.guideline_level == "extensive":
            return EXTENSIVE_GUIDELINE
        elif self.guideline_level == "detailed":
            return DETAILED_GUIDELINE
        elif self.guideline_level == "detailed_hierarchical":
            return DETAILED_GUIDELINE_HIERARCHICAL
        elif self.guideline_level == "detailed_flat":
            return DETAILED_GUIDELINE_FLAT
        elif self.guideline_level == "detailed_flat_turn7":
            return DETAILED_GUIDELINE_FLAT_TURN7
        elif self.guideline_level == "detailed_flat_multiTQ":
            return DETAILED_GUIDELINE_FLAT_MULTITQ
        elif self.guideline_level == "detailed_flat_multiTQ_v2":
            return DETAILED_GUIDELINE_FLAT_MULTITQ_V2
        elif self.guideline_level == "detailed_flat_multiTQ_v2_q5":
            return DETAILED_GUIDELINE_FLAT_MULTITQ_V2_Q5
        elif self.guideline_level == "detailed_flat_multiTQ_v3":
            return DETAILED_GUIDELINE_FLAT_MULTITQ_V3
        elif self.guideline_level == "detailed_flat_10arxiv":
            return DETAILED_GUIDELINE_FLAT_10ARXIV
        elif self.guideline_level == "detailed_minimal":
            return DETAILED_GUIDELINE_MINIMAL
        elif self.guideline_level == "detailed_minimal_5q":
            return DETAILED_GUIDELINE_MINIMAL_5Q
        elif self.guideline_level == "vanilla":
            return VANILLA_GUIDELINE
        elif self.guideline_level == "cot":
            return COT_GUIDELINE
        else:
            return ""
    
    def should_apply_hints(self) -> bool:
        """Check if hints should be applied based on current configuration."""
        if not self.enable:
            return False
        
        # Check if we should add hints based on current step
        if self.hint_steps > 0 and self.current_step >= self.hint_steps:
            return False
            
        return True
    
    def augment_prompt(self, base_prompt: str) -> str:
        """
        Apply prompt augmentation with instruction hints.
        
        For vanilla mode, this extracts clean questions and creates simple prompts.
        For other modes, this applies KG instruction hints as usual.
        
        Args:
            base_prompt: The original prompt text
            
        Returns:
            Augmented prompt with instruction hints if enabled, otherwise original prompt
        """
        if not self.should_apply_hints():
            return base_prompt
        
        # Special handling for vanilla and COT modes: extract clean question and create appropriate prompt
        if self.guideline_level == "vanilla":
            # Extract clean question from the KG-augmented prompt
            clean_question = extract_clean_question_from_prompt(base_prompt)
            
            # Create a simple vanilla prompt
            vanilla_prompt = create_vanilla_prompt(clean_question)
            
            # Removed debug logging for extracted questions
            return vanilla_prompt
        
        elif self.guideline_level == "cot":
            # Extract clean question from the KG-augmented prompt
            clean_question = extract_clean_question_from_prompt(base_prompt)
            
            # Create a COT prompt that encourages reasoning
            cot_prompt = create_cot_prompt(clean_question)
            
            # Removed debug logging for extracted questions
            return cot_prompt
        
        # Regular KG mode augmentation
        instruction_hint = self.get_instruction_hint()
        if not instruction_hint:
            return base_prompt
        
        # Try to find question pattern to insert hint before the question
        question_match = None
        for pattern in self.question_patterns:
            question_match = pattern.search(base_prompt)
            if question_match:
                break
        
        if question_match:
            # Extract the part before the question and the question itself
            question_start = question_match.start()
            prompt_before_question = base_prompt[:question_start]
            question_part = base_prompt[question_start:]
            
            # Insert hint between prompt content and question
            augmented_prompt = f"{prompt_before_question}\n\n[Hint]: {instruction_hint}\n\n{question_part}"
            return augmented_prompt
        else:
            # Fallback: if no question pattern found, append hint at the end
            augmented_prompt = f"{base_prompt}\n\n[Hint]: {instruction_hint}"
            return augmented_prompt


def create_prompt_augmentor_from_config(config) -> PromptAugmentor:
    """
    Create a PromptAugmentor instance from configuration.
    
    Args:
        config: Configuration object with prompt_augmentation section
        
    Returns:
        PromptAugmentor instance
    """
    augmentation_config = config.get("prompt_augmentation", {})
    
    return PromptAugmentor(
        enable=augmentation_config.get("enable", False),
        guideline_level=augmentation_config.get("guideline_level", None),
        hint_steps=augmentation_config.get("hint_steps", 0),
        current_step=0  # Will be updated by trainer
    )
