"""
Control Flow Node Executors - Branching, loops, and flow control
"""

from typing import Dict, Any, List
import asyncio
from .base import NodeExecutor, ExecutionContext


class ConditionExecutor(NodeExecutor):
    """Branch based on condition"""
    
    node_type = "condition"
    display_name = "Condition"
    category = "control"
    description = "Branch execution based on a condition"
    
    # Special marker for control flow
    is_control_flow = True
    
    async def execute(self, inputs: Dict[str, Any], context: ExecutionContext) -> Dict[str, Any]:
        input_value = inputs.get("input")
        condition = self.get_config_value("condition", "True")
        
        context.log(f"Evaluating condition: {condition}")
        
        try:
            # Create evaluation context
            local_vars = {
                "input": input_value,
                "value": input_value,
                **context.variables,
            }
            
            # Safe builtins
            safe_builtins = {
                "len": len,
                "str": str,
                "int": int,
                "float": float,
                "bool": bool,
                "list": list,
                "dict": dict,
                "abs": abs,
                "min": min,
                "max": max,
                "sum": sum,
                "any": any,
                "all": all,
                "True": True,
                "False": False,
                "None": None,
            }
            
            result = eval(condition, {"__builtins__": safe_builtins}, local_vars)
            is_true = bool(result)
            
            context.log(f"Condition result: {is_true}")
            
            # Return to both branches, engine will handle routing
            return {
                "true": input_value if is_true else None,
                "false": input_value if not is_true else None,
                "_condition_result": is_true,
            }
        except Exception as e:
            context.log(f"Condition evaluation error: {e}", level="error")
            raise ValueError(f"Condition evaluation failed: {e}")


class SwitchExecutor(NodeExecutor):
    """Multi-way branch"""
    
    node_type = "switch"
    display_name = "Switch"
    category = "control"
    description = "Route to different outputs based on value"
    
    is_control_flow = True
    
    async def execute(self, inputs: Dict[str, Any], context: ExecutionContext) -> Dict[str, Any]:
        input_value = inputs.get("input")
        cases = self.get_config_value("cases", [])
        
        context.log(f"Switch on value: {input_value}")
        
        outputs = {"default": None}
        matched = False
        
        for case in cases:
            case_value = case.get("value")
            case_label = case.get("label", str(case_value))
            
            if str(input_value) == str(case_value):
                outputs[case_label] = input_value
                matched = True
                context.log(f"Matched case: {case_label}")
                break
        
        if not matched:
            outputs["default"] = input_value
            context.log("Using default case")
        
        return outputs


class LoopExecutor(NodeExecutor):
    """Iterate over items"""
    
    node_type = "loop"
    display_name = "Loop"
    category = "control"
    description = "Iterate over array items"
    
    is_control_flow = True
    
    async def execute(self, inputs: Dict[str, Any], context: ExecutionContext) -> Dict[str, Any]:
        items = inputs.get("items", [])
        max_iterations = self.get_config_value("maxIterations", 100)
        
        if not items:
            context.log("No items to iterate")
            return {"item": None, "index": -1, "done": []}
        
        # Get current iteration from context
        loop_state_key = f"_loop_{id(self)}"
        loop_state = context.variables.get(loop_state_key, {"index": 0, "results": []})
        
        current_index = loop_state["index"]
        
        if current_index >= len(items) or current_index >= max_iterations:
            # Loop complete
            context.log(f"Loop complete after {current_index} iterations")
            del context.variables[loop_state_key]
            return {
                "item": None,
                "index": current_index,
                "done": loop_state["results"],
            }
        
        # Get current item
        current_item = items[current_index]
        context.log(f"Loop iteration {current_index + 1}/{len(items)}")
        
        # Update state for next iteration
        loop_state["index"] = current_index + 1
        context.variables[loop_state_key] = loop_state
        
        return {
            "item": current_item,
            "index": current_index,
            "done": None,  # Not done yet
        }


class MergeExecutor(NodeExecutor):
    """Merge multiple inputs"""
    
    node_type = "merge"
    display_name = "Merge"
    category = "control"
    description = "Combine multiple branch outputs"
    
    async def execute(self, inputs: Dict[str, Any], context: ExecutionContext) -> Dict[str, Any]:
        # Collect all non-None inputs
        results = []
        
        for key in ["input1", "input2", "input3"]:
            value = inputs.get(key)
            if value is not None:
                results.append(value)
        
        # Also check for any other inputs
        for key, value in inputs.items():
            if key not in ["input1", "input2", "input3"] and value is not None:
                results.append(value)
        
        context.log(f"Merged {len(results)} inputs")
        return {"output": results}


class DelayExecutor(NodeExecutor):
    """Wait for specified duration"""
    
    node_type = "delay"
    display_name = "Delay"
    category = "control"
    description = "Pause execution for a specified time"
    
    async def execute(self, inputs: Dict[str, Any], context: ExecutionContext) -> Dict[str, Any]:
        input_value = inputs.get("input")
        ms = self.get_config_value("ms", 1000)
        
        context.log(f"Delaying for {ms}ms")
        await asyncio.sleep(ms / 1000)
        
        return {"output": input_value}


class RetryExecutor(NodeExecutor):
    """Retry on failure"""
    
    node_type = "retry"
    display_name = "Retry"
    category = "control"
    description = "Retry failed operations"
    
    async def execute(self, inputs: Dict[str, Any], context: ExecutionContext) -> Dict[str, Any]:
        input_value = inputs.get("input")
        max_retries = self.get_config_value("maxRetries", 3)
        delay_ms = self.get_config_value("delayMs", 1000)
        
        # This executor just passes through - retry logic is handled by the engine
        context.log(f"Retry wrapper: max {max_retries} retries with {delay_ms}ms delay")
        
        return {"output": input_value}


class ParallelExecutor(NodeExecutor):
    """Execute branches in parallel"""
    
    node_type = "parallel"
    display_name = "Parallel"
    category = "control"
    description = "Execute multiple branches simultaneously"
    
    is_control_flow = True
    
    async def execute(self, inputs: Dict[str, Any], context: ExecutionContext) -> Dict[str, Any]:
        # Parallel execution is handled by the engine
        # This node just distributes the input
        input_value = inputs.get("input")
        
        context.log("Starting parallel execution")
        
        return {
            "branch1": input_value,
            "branch2": input_value,
            "branch3": input_value,
        }
