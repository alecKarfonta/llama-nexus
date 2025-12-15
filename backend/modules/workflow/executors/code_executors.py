"""
Code Execution Node Executors
Execute Python and JavaScript code safely
"""

from typing import Dict, Any, List
import sys
import io
import ast
import json
import traceback
from contextlib import redirect_stdout, redirect_stderr
from .base import NodeExecutor, ExecutionContext


class CodeExecutorExecutor(NodeExecutor):
    """Execute Python or JavaScript code"""
    
    node_type = "code_executor"
    display_name = "Code Executor"
    category = "tools"
    description = "Execute Python or JavaScript code"
    
    # Allowed built-ins for safe execution
    SAFE_BUILTINS = {
        'abs': abs,
        'all': all,
        'any': any,
        'bool': bool,
        'dict': dict,
        'enumerate': enumerate,
        'filter': filter,
        'float': float,
        'int': int,
        'len': len,
        'list': list,
        'map': map,
        'max': max,
        'min': min,
        'print': print,
        'range': range,
        'reversed': reversed,
        'round': round,
        'set': set,
        'sorted': sorted,
        'str': str,
        'sum': sum,
        'tuple': tuple,
        'type': type,
        'zip': zip,
        'True': True,
        'False': False,
        'None': None,
    }
    
    async def execute(self, inputs: Dict[str, Any], context: ExecutionContext) -> Dict[str, Any]:
        input_value = inputs.get("input")
        
        language = self.get_config_value("language", "python")
        code = self.get_config_value("code", "")
        timeout = self.get_config_value("timeout", 30)
        
        if not code:
            raise ValueError("Code is required")
        
        context.log(f"Executing {language} code")
        
        if language == "python":
            return await self._execute_python(code, input_value, context, timeout)
        elif language == "javascript":
            return await self._execute_javascript(code, input_value, context)
        else:
            raise ValueError(f"Unsupported language: {language}")
    
    async def _execute_python(
        self, 
        code: str, 
        input_value: Any, 
        context: ExecutionContext,
        timeout: int
    ) -> Dict[str, Any]:
        """Execute Python code in a sandboxed environment"""
        
        # Capture stdout and stderr
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        
        # Create safe execution environment
        safe_globals = {
            '__builtins__': self.SAFE_BUILTINS,
            'input': input_value,
            'json': json,
        }
        
        # Add commonly used modules (safe subset)
        try:
            import re
            safe_globals['re'] = re
        except:
            pass
        
        try:
            import math
            safe_globals['math'] = math
        except:
            pass
        
        try:
            import datetime
            safe_globals['datetime'] = datetime
        except:
            pass
        
        # Local namespace for results
        local_vars = {
            'result': None,
        }
        
        try:
            # Parse the code to check for dangerous operations
            tree = ast.parse(code)
            
            # Check for dangerous imports
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name in ['os', 'sys', 'subprocess', 'shutil', 'socket']:
                            raise ValueError(f"Import of {alias.name} is not allowed")
                elif isinstance(node, ast.ImportFrom):
                    if node.module in ['os', 'sys', 'subprocess', 'shutil', 'socket']:
                        raise ValueError(f"Import from {node.module} is not allowed")
            
            # Execute with captured output
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                exec(code, safe_globals, local_vars)
            
            result = local_vars.get('result')
            stdout = stdout_capture.getvalue()
            stderr = stderr_capture.getvalue()
            
            context.log(f"Python execution completed")
            
            return {
                "result": result,
                "stdout": stdout,
                "stderr": stderr,
            }
            
        except SyntaxError as e:
            context.log(f"Python syntax error: {e}", level="error")
            raise ValueError(f"Syntax error: {e}")
        except Exception as e:
            context.log(f"Python execution error: {e}", level="error")
            raise ValueError(f"Execution error: {e}")
    
    async def _execute_javascript(
        self, 
        code: str, 
        input_value: Any, 
        context: ExecutionContext
    ) -> Dict[str, Any]:
        """Execute JavaScript code (requires Node.js or js2py)"""
        
        # Try using js2py for simple JavaScript
        try:
            import js2py
        except ImportError:
            raise ValueError("JavaScript execution requires js2py. Run: pip install js2py")
        
        try:
            # Create JavaScript context
            js_context = js2py.EvalJs()
            
            # Set input variable
            js_context.input = input_value
            
            # Capture console.log output
            logs = []
            js_context.execute("""
                var _logs = [];
                var console = {
                    log: function() {
                        _logs.push(Array.prototype.slice.call(arguments).join(' '));
                    }
                };
            """)
            
            # Execute the code
            result = js_context.execute(code)
            
            # Get logs
            stdout = '\n'.join(js_context._logs.to_list())
            
            context.log(f"JavaScript execution completed")
            
            return {
                "result": result,
                "stdout": stdout,
                "stderr": "",
            }
            
        except Exception as e:
            context.log(f"JavaScript execution error: {e}", level="error")
            raise ValueError(f"JavaScript execution error: {e}")


class FunctionCallExecutor(NodeExecutor):
    """Call a registered function/tool"""
    
    node_type = "function_call"
    display_name = "Function Call"
    category = "tools"
    description = "Call a registered function by name"
    
    async def execute(self, inputs: Dict[str, Any], context: ExecutionContext) -> Dict[str, Any]:
        args = inputs.get("args", {})
        
        function_name = self.get_config_value("functionName", "")
        
        if not function_name:
            raise ValueError("Function name is required")
        
        context.log(f"Calling function: {function_name}")
        
        # Check for built-in functions
        builtin_functions = {
            'json_encode': lambda x: json.dumps(x, default=str),
            'json_decode': lambda x: json.loads(x),
            'to_string': lambda x: str(x),
            'to_int': lambda x: int(x),
            'to_float': lambda x: float(x),
            'to_bool': lambda x: bool(x),
            'length': lambda x: len(x),
            'keys': lambda x: list(x.keys()) if isinstance(x, dict) else [],
            'values': lambda x: list(x.values()) if isinstance(x, dict) else [],
            'first': lambda x: x[0] if x else None,
            'last': lambda x: x[-1] if x else None,
            'reverse': lambda x: list(reversed(x)) if isinstance(x, (list, tuple)) else x,
            'sort': lambda x: sorted(x) if isinstance(x, (list, tuple)) else x,
            'unique': lambda x: list(set(x)) if isinstance(x, (list, tuple)) else x,
            'flatten': self._flatten,
            'merge_dicts': lambda *dicts: {k: v for d in dicts for k, v in d.items()},
        }
        
        if function_name in builtin_functions:
            try:
                if isinstance(args, dict):
                    result = builtin_functions[function_name](**args)
                elif isinstance(args, (list, tuple)):
                    result = builtin_functions[function_name](*args)
                else:
                    result = builtin_functions[function_name](args)
                
                return {"result": result}
            except Exception as e:
                context.log(f"Function call error: {e}", level="error")
                raise ValueError(f"Function '{function_name}' error: {e}")
        
        # Check for custom functions in context
        custom_functions = context.variables.get("_functions", {})
        if function_name in custom_functions:
            try:
                func = custom_functions[function_name]
                result = func(args) if callable(func) else None
                return {"result": result}
            except Exception as e:
                context.log(f"Custom function error: {e}", level="error")
                raise ValueError(f"Function '{function_name}' error: {e}")
        
        raise ValueError(f"Unknown function: {function_name}")
    
    def _flatten(self, lst: list, depth: int = 1) -> list:
        """Flatten a nested list"""
        result = []
        for item in lst:
            if isinstance(item, list) and depth > 0:
                result.extend(self._flatten(item, depth - 1))
            else:
                result.append(item)
        return result


class ShellCommandExecutor(NodeExecutor):
    """Execute shell command (restricted for safety)"""
    
    node_type = "shell_command"
    display_name = "Shell Command"
    category = "tools"
    description = "Execute a shell command (restricted)"
    
    # Only allow these safe commands
    ALLOWED_COMMANDS = {
        'echo', 'date', 'whoami', 'pwd', 'ls', 'cat', 'head', 'tail',
        'wc', 'sort', 'uniq', 'grep', 'awk', 'sed', 'cut', 'tr',
        'curl', 'wget', 'jq', 'base64', 'md5sum', 'sha256sum',
    }
    
    async def execute(self, inputs: Dict[str, Any], context: ExecutionContext) -> Dict[str, Any]:
        import asyncio
        import shlex
        
        command = self.get_config_value("command", "")
        timeout = self.get_config_value("timeout", 30)
        
        if not command:
            raise ValueError("Command is required")
        
        # Parse command to check safety
        try:
            parts = shlex.split(command)
            base_cmd = parts[0] if parts else ""
        except ValueError as e:
            raise ValueError(f"Invalid command syntax: {e}")
        
        # Check if command is allowed
        if base_cmd not in self.ALLOWED_COMMANDS:
            raise ValueError(
                f"Command '{base_cmd}' is not allowed. "
                f"Allowed commands: {', '.join(sorted(self.ALLOWED_COMMANDS))}"
            )
        
        context.log(f"Executing shell command: {base_cmd}")
        
        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout
                )
            except asyncio.TimeoutError:
                process.kill()
                raise ValueError(f"Command timed out after {timeout}s")
            
            return {
                "stdout": stdout.decode('utf-8', errors='replace'),
                "stderr": stderr.decode('utf-8', errors='replace'),
                "code": process.returncode,
            }
            
        except Exception as e:
            context.log(f"Shell command error: {e}", level="error")
            raise ValueError(f"Shell command error: {e}")










