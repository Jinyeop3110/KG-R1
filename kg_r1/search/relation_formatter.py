"""
Relation formatting utilities for hierarchical representation of Freebase relations.

This module provides functions to convert flat Freebase relation lists into 
hierarchical structures for improved token efficiency and readability.
"""

import os
from collections import defaultdict
from typing import List, Dict, Any
from enum import Enum


class RelationFormat(str, Enum):
    """Supported relation formatting types."""
    FLAT = "flat"                    # Current format: rel1, rel2, rel3
    FULL_INDENTATION = "full_indent" # Full hierarchy with indentation
    MIXED = "mixed"                  # Mixed hierarchy with comma-separated properties
    COMPACT = "compact"              # Compact domain.type: props format


def get_relation_format_from_env() -> RelationFormat:
    """Get relation format from environment variable, default to full indentation."""
    format_str = os.environ.get('KG_RELATION_FORMAT', 'full_indent').lower()
    
    try:
        return RelationFormat(format_str)
    except ValueError:
        # Invalid format, default to full indentation
        return RelationFormat.FULL_INDENTATION


def group_relations_hierarchically(relations: List[str]) -> Dict[str, Dict[str, List[str]]]:
    """
    Group flat Freebase relations into hierarchical structure.
    
    Handles relations with different numbers of components:
    - people.person.nationality → domain: people, type: person, property: nationality
    - base.gender.personal_gender_identity.gender_identity → domain: base.gender, type: personal_gender_identity, property: gender_identity
    - base.locations.countries.planet → domain: base.locations, type: countries, property: planet
    
    Args:
        relations: List of Freebase relations like ["people.person.nationality", "base.gender.personal_gender_identity.gender_identity"]
        
    Returns:
        Hierarchical structure: {domain: {type: [properties]}}
    """
    grouped = defaultdict(lambda: defaultdict(list))
    
    for relation in relations:
        # Handle empty or malformed relations
        if not relation or not isinstance(relation, str):
            continue
            
        parts = relation.split('.')
        if len(parts) >= 4:
            # For 4+ part relations, use first 2 parts as domain
            # e.g., base.gender.personal_gender_identity.gender_identity
            # → domain: "base.gender", type: "personal_gender_identity", property: "gender_identity"
            domain = f"{parts[0]}.{parts[1]}"  # First two parts as domain
            type_name = parts[2]               # Third part as type
            
            # Join remaining parts as property
            if len(parts) > 4:
                property_name = '.'.join(parts[3:])  # Join remaining parts
            else:
                property_name = parts[3]  # Fourth part as property
                
            grouped[domain][type_name].append(property_name)
        elif len(parts) == 3:
            # Standard 3-part relation
            # e.g., people.person.nationality → domain: "people", type: "person", property: "nationality"  
            domain = parts[0]
            type_name = parts[1]
            property_name = parts[2]
            
            grouped[domain][type_name].append(property_name)
        else:
            # Handle malformed relations (< 3 parts) - put in special category
            grouped["other"]["unknown"].append(relation)
    
    return dict(grouped)


def format_relations_flat(relations: List[str]) -> str:
    """
    Format relations in current flat format.
    
    Args:
        relations: List of relation strings
        
    Returns:
        Comma-separated relation string
    """
    return ', '.join(relations)


def format_relations_full_indentation(relations: List[str]) -> str:
    """
    Format relations with full indentation hierarchy.
    
    Example output:
    people
      person
        nationality
        spouse_s
    location
      country
        capital
        
    Args:
        relations: List of relation strings
        
    Returns:
        Hierarchically indented relation string
    """
    grouped = group_relations_hierarchically(relations)
    
    result = []
    for domain in sorted(grouped.keys()):
        result.append(domain)
        for type_name in sorted(grouped[domain].keys()):
            result.append(f"  {type_name}")
            for prop in sorted(grouped[domain][type_name]):
                result.append(f"    {prop}")
    
    return "\n".join(result)


def format_relations_mixed(relations: List[str]) -> str:
    """
    Format relations with mixed indentation (properties on same line).
    
    Example output:
    people
      person: nationality, spouse_s
    location
      country: capital
        
    Args:
        relations: List of relation strings
        
    Returns:
        Mixed hierarchy relation string
    """
    grouped = group_relations_hierarchically(relations)
    
    result = []
    for domain in sorted(grouped.keys()):
        result.append(domain)
        for type_name in sorted(grouped[domain].keys()):
            properties = ", ".join(sorted(grouped[domain][type_name]))
            result.append(f"  {type_name}: {properties}")
    
    return "\n".join(result)


def format_relations_compact(relations: List[str]) -> str:
    """
    Format relations in compact domain.type: properties format.
    
    Example output:
    people.person: nationality, spouse_s
    location.country: capital
        
    Args:
        relations: List of relation strings
        
    Returns:
        Compact relation string
    """
    grouped = group_relations_hierarchically(relations)
    
    result = []
    for domain in sorted(grouped.keys()):
        for type_name in sorted(grouped[domain].keys()):
            properties = ", ".join(sorted(grouped[domain][type_name]))
            result.append(f"{domain}.{type_name}: {properties}")
    
    return "\n".join(result)


def format_relations(relations: List[str], format_type: RelationFormat = None) -> str:
    """
    Format relations according to specified format type.
    
    Args:
        relations: List of relation strings
        format_type: Desired formatting type (defaults to environment setting)
        
    Returns:
        Formatted relation string
    """
    if format_type is None:
        format_type = get_relation_format_from_env()
    
    if format_type == RelationFormat.FLAT:
        return format_relations_flat(relations)
    elif format_type == RelationFormat.FULL_INDENTATION:
        return format_relations_full_indentation(relations)
    elif format_type == RelationFormat.MIXED:
        return format_relations_mixed(relations)  
    elif format_type == RelationFormat.COMPACT:
        return format_relations_compact(relations)
    else:
        # Default to full indentation for unknown formats
        return format_relations_full_indentation(relations)


def estimate_token_savings(relations: List[str], target_format: RelationFormat = RelationFormat.FULL_INDENTATION) -> Dict[str, Any]:
    """
    Estimate token savings from format conversion.
    
    Args:
        relations: List of relation strings
        target_format: Target format to compare against
        
    Returns:
        Dictionary with savings estimates
    """
    if not relations:
        return {
            'current_length': 0,
            'new_length': 0,
            'savings_chars': 0,
            'savings_percent': 0.0
        }
    
    current_format = format_relations_flat(relations)
    new_format = format_relations(relations, target_format)
    
    current_length = len(current_format)
    new_length = len(new_format)
    savings_chars = current_length - new_length
    savings_percent = (savings_chars / current_length * 100) if current_length > 0 else 0.0
    
    return {
        'current_length': current_length,
        'new_length': new_length,
        'savings_chars': savings_chars,
        'savings_percent': savings_percent,
        'relation_count': len(relations)
    }