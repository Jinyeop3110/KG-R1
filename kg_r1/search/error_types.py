"""
Action-specific error classification system for KG reward calculation tracking.

This module provides a precise error classification system where each action handler
explicitly returns the appropriate error type based on the specific failure case.
No string matching - each error is determined by the code logic itself.

Error Types by Action:
- SERVER_ERROR: Server/system level issues (unsupported dataset, no actions enabled, etc.)
- SAMPLE_NOT_FOUND: Sample/subgraph not found in KG
- ENTITY_NOT_FOUND: Entity not found in the specific sample's subgraph
- RELATION_NOT_FOUND: Relation not found in KG
- NO_RESULTS: Valid query but no results (empty result set)
- FORMAT_ERROR: Invalid request format, missing required fields
- SUCCESS: Operation completed successfully

Usage:
    from error_types import KGErrorType
    
    # In action handlers, explicitly return error type
    if sample_id not in self.kg.subgraphs:
        return kg_retrieval_completion_response(
            f'Sample "{sample_id}" not found in KG',
            "get_relations", 
            is_error=True, 
            error_type=KGErrorType.SAMPLE_NOT_FOUND
        )
"""

class KGErrorType:
    """Precise error types for KG operations with explicit action-based classification."""
    
    # System/Server level errors
    SERVER_ERROR = "KG_SERVER_ERROR"           # Unsupported dataset, no actions enabled, HTTP errors
    FORMAT_ERROR = "KG_FORMAT_ERROR"           # Invalid request format, missing required fields
    
    # Data not found errors  
    SAMPLE_NOT_FOUND = "KG_SAMPLE_NOT_FOUND"   # Sample/subgraph not found
    ENTITY_NOT_FOUND = "KG_ENTITY_NOT_FOUND"   # Entity not found in sample's subgraph
    RELATION_NOT_FOUND = "KG_RELATION_NOT_FOUND" # Relation not found in KG
    
    # Valid query but no results
    NO_RESULTS = "KG_NO_RESULTS"               # Empty result set for valid query
    
    # Success
    SUCCESS = "KG_SUCCESS"                     # Operation completed successfully
    
    @classmethod
    def get_all_types(cls):
        """Get all error types as a list."""
        return [
            cls.SERVER_ERROR,
            cls.FORMAT_ERROR, 
            cls.SAMPLE_NOT_FOUND,
            cls.ENTITY_NOT_FOUND,
            cls.RELATION_NOT_FOUND,
            cls.NO_RESULTS,
            cls.SUCCESS
        ]
    
    @classmethod
    def get_error_descriptions(cls) -> dict:
        """Get detailed descriptions for each error type."""
        return {
            cls.SERVER_ERROR: "Server/system level issues (unsupported dataset, service unavailable)",
            cls.FORMAT_ERROR: "Invalid request format or missing required fields",
            cls.SAMPLE_NOT_FOUND: "Sample/subgraph not found in knowledge graph",
            cls.ENTITY_NOT_FOUND: "Entity not found in the sample's subgraph",
            cls.RELATION_NOT_FOUND: "Relation not found in knowledge graph",
            cls.NO_RESULTS: "Valid query but no results found",
            cls.SUCCESS: "Operation completed successfully"
        }
    
    @classmethod
    def is_error_type(cls, error_type: str) -> bool:
        """Check if the given type represents an error (not success)."""
        return error_type != cls.SUCCESS
    
    @classmethod
    def get_error_stats_template(cls) -> dict:
        """Get a template dictionary for error statistics tracking."""
        return {error_type: 0 for error_type in cls.get_all_types()}

def get_error_analytics(error_counts: dict) -> dict:
    """Generate analytics summary from error counts."""
    total_operations = sum(error_counts.values())
    total_errors = sum(count for error_type, count in error_counts.items() 
                      if KGErrorType.is_error_type(error_type))
    
    analytics = {
        "total_operations": total_operations,
        "total_errors": total_errors,
        "total_successes": error_counts.get(KGErrorType.SUCCESS, 0),
        "error_rate": (total_errors / total_operations) if total_operations > 0 else 0,
        "success_rate": (error_counts.get(KGErrorType.SUCCESS, 0) / total_operations) if total_operations > 0 else 0,
        "error_distribution": {},
        "most_common_error": None
    }
    
    if total_errors > 0:
        # Calculate distribution percentages for errors only
        error_only_counts = {k: v for k, v in error_counts.items() if KGErrorType.is_error_type(k)}
        for error_type, count in error_only_counts.items():
            analytics["error_distribution"][error_type] = {
                "count": count,
                "percentage": round((count / total_errors) * 100, 2)
            }
        
        # Find most common error (excluding success)
        analytics["most_common_error"] = max(error_only_counts, key=error_only_counts.get)
    
    return analytics

def validate_error_tracking(error_counts: dict) -> bool:
    """Validate that error counts contain all required error types."""
    expected_types = set(KGErrorType.get_all_types())
    actual_types = set(error_counts.keys())
    return expected_types.issubset(actual_types)

def merge_error_counts(counts1: dict, counts2: dict) -> dict:
    """Merge two error count dictionaries."""
    merged = KGErrorType.get_error_stats_template()
    for error_type in merged.keys():
        merged[error_type] = counts1.get(error_type, 0) + counts2.get(error_type, 0)
    return merged

# Action-specific error mapping for reference
ACTION_ERROR_MAPPING = {
    "get_relations": {
        "sample_not_found": KGErrorType.SAMPLE_NOT_FOUND,
        "entity_not_found": KGErrorType.ENTITY_NOT_FOUND, 
        "no_relations": KGErrorType.NO_RESULTS,
        "success": KGErrorType.SUCCESS
    },
    "get_head_entities": {
        "sample_not_found": KGErrorType.SAMPLE_NOT_FOUND,
        "entity_not_found": KGErrorType.ENTITY_NOT_FOUND,
        "relation_not_found": KGErrorType.RELATION_NOT_FOUND,
        "no_entities": KGErrorType.NO_RESULTS,
        "success": KGErrorType.SUCCESS
    },
    "get_tail_entities": {
        "sample_not_found": KGErrorType.SAMPLE_NOT_FOUND,
        "entity_not_found": KGErrorType.ENTITY_NOT_FOUND,
        "relation_not_found": KGErrorType.RELATION_NOT_FOUND,
        "no_entities": KGErrorType.NO_RESULTS,
        "success": KGErrorType.SUCCESS
    },
    "server": {
        "unsupported_dataset": KGErrorType.SERVER_ERROR,
        "unsupported_action": KGErrorType.SERVER_ERROR,
        "no_actions_enabled": KGErrorType.SERVER_ERROR,
        "validation_error": KGErrorType.FORMAT_ERROR
    }
}
