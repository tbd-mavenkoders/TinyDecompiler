"""
ASAN Error Parser for LLM Prompting
"""

import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class StackFrame:
    """
    Represents a single stack frame.
    """
    frame_num: int
    address: str
    function: str
    file: str
    line: Optional[int] = None
    column: Optional[int] = None
    
    def __str__(self) -> str:
        location = f"{self.file}"
        if self.line:
            location += f":{self.line}"
            if self.column:
                location += f":{self.column}"
        return f"#{self.frame_num} {self.function} at {location}"


@dataclass
class AsanError:
    """
    Structured representation of an ASAN error.
    """
    error_type: str = ""
    operation: str = ""  # READ, WRITE, etc.
    size: Optional[int] = None
    address: str = ""
    file: str = ""
    line: Optional[int] = None
    column: Optional[int] = None
    stack_trace: List[StackFrame] = field(default_factory=list)
    allocation_trace: List[StackFrame] = field(default_factory=list)
    memory_info: str = ""
    summary: str = ""
    raw_output: str = ""
    
    def is_valid(self) -> bool:
        return bool(self.error_type and (self.file or self.stack_trace))


class AsanParser:
    """
    Robust parser for AddressSanitizer error output.
    """
    
    # Common ASAN error patterns
    ERROR_PATTERNS = [
        r'ERROR: AddressSanitizer: ([^\s]+)',
        r'ERROR: LeakSanitizer: (.+)',
    ]
    
    OPERATION_PATTERN = r'(READ|WRITE|ACCESS) of size (\d+)'
    LOCATION_PATTERN = r'([^\s]+\.(?:c|cpp|cc|cxx|h|hpp)):(\d+)(?::(\d+))?'
    STACK_FRAME_PATTERN = r'#(\d+)\s+0x([0-9a-f]+)\s+(?:in\s+)?([^\s]+)\s+(.+)'
    ADDRESS_PATTERN = r'0x[0-9a-f]+'
    
    def __init__(self, asan_output: str):
        """
        Initialize parser with ASAN output.
        """
        self.raw_output = asan_output.strip()
        self.error = AsanError(raw_output=self.raw_output)
        self.parse()
    
    def parse(self):
        """
        Parse the ASAN output and extract all relevant information.
        """
        if not self.raw_output:
            return
        
        lines = self.raw_output.split('\n')
        
        in_stack_trace = False
        in_allocation_trace = False
        
        for i, line in enumerate(lines):
            line = line.strip()
            
            # Parse error type
            if not self.error.error_type:
                for pattern in self.ERROR_PATTERNS:
                    match = re.search(pattern, line)
                    if match:
                        self.error.error_type = match.group(1).strip()
                        break
            
            # Parse operation and size
            if not self.error.operation:
                match = re.search(self.OPERATION_PATTERN, line, re.IGNORECASE)
                if match:
                    self.error.operation = match.group(1)
                    self.error.size = int(match.group(2))
            
            # Parse address
            if not self.error.address and self.error.error_type:
                addresses = re.findall(self.ADDRESS_PATTERN, line)
                if addresses:
                    self.error.address = addresses[0]
            
            # Parse primary error location
            if not self.error.file and self.error.error_type:
                match = re.search(self.LOCATION_PATTERN, line)
                if match:
                    self.error.file = match.group(1)
                    self.error.line = int(match.group(2))
                    if match.group(3):
                        self.error.column = int(match.group(3))
            
            # Detect stack trace sections
            if 'allocated by thread' in line.lower() or 'previously allocated' in line.lower():
                in_allocation_trace = True
                in_stack_trace = False
                continue
            
            # Parse stack frames
            if line.startswith('#'):
                frame = self.parse_stack_frame(line)
                if frame:
                    if in_allocation_trace:
                        self.error.allocation_trace.append(frame)
                    else:
                        in_stack_trace = True
                        self.error.stack_trace.append(frame)
            else:
                # Stop stack trace on non-frame lines (unless it's an empty line)
                if in_stack_trace and line and not line.startswith('=='):
                    if not any(keyword in line.lower() for keyword in 
                              ['allocated', 'freed', 'address', 'thread', 'shadow']):
                        in_stack_trace = False
            
            # Parse memory information
            if 'is located' in line or 'bytes' in line.lower():
                if not self.error.memory_info and line and not line.startswith('#'):
                    self.error.memory_info = line
            
            # Parse summary
            if 'SUMMARY:' in line:
                self.error.summary = line.replace('SUMMARY:', '').strip()
    
    def parse_stack_frame(self, line: str) -> Optional[StackFrame]:
        """
        Parse a single stack frame line.
        Args:
            line: A line containing stack frame information
            
        Returns:
            StackFrame object or None if parsing fails
        """
        # Try full pattern with location
        match = re.search(self.STACK_FRAME_PATTERN, line)
        if match:
            frame_num = int(match.group(1))
            address = match.group(2)
            function = match.group(3)
            location_str = match.group(4)
            
            # Parse location (file:line:column or file:line or just file)
            loc_match = re.search(self.LOCATION_PATTERN, location_str)
            if loc_match:
                file = loc_match.group(1)
                line_num = int(loc_match.group(2)) if loc_match.group(2) else None
                col_num = int(loc_match.group(3)) if loc_match.group(3) else None
                
                return StackFrame(
                    frame_num=frame_num,
                    address=address,
                    function=function,
                    file=file,
                    line=line_num,
                    column=col_num
                )
            else:
                # No location info, just file path or binary
                return StackFrame(
                    frame_num=frame_num,
                    address=address,
                    function=function,
                    file=location_str.strip()
                )
        
        return None
    
    def format_for_llm(self, include_raw: bool = False, max_stack_depth: int = 10) -> str:
        """
        Format the parsed error for LLM consumption.
        
        Args:
            include_raw: Whether to include raw ASAN output at the end
            max_stack_depth: Maximum number of stack frames to include
            
        Returns:
            Formatted string optimized for LLM understanding
        """
        if not self.error.is_valid():
            # Fallback to raw output if parsing failed
            return f"ASAN Error (parsing incomplete):\n{self.raw_output}"
        
        sections = []
        
        # Header
        sections.append("=" * 60)
        sections.append("ADDRESSSANITIZER ERROR REPORT")
        sections.append("=" * 60)
        
        # Error type and operation
        error_desc = f"Error Type: {self.error.error_type}"
        if self.error.operation:
            error_desc += f" ({self.error.operation}"
            if self.error.size:
                error_desc += f" of {self.error.size} byte{'s' if self.error.size > 1 else ''}"
            error_desc += ")"
        sections.append(error_desc)
        
        # Primary location
        if self.error.file:
            location = f"Location: {self.error.file}"
            if self.error.line:
                location += f":{self.error.line}"
                if self.error.column:
                    location += f":{self.error.column}"
            sections.append(location)
        
        # Address
        if self.error.address:
            sections.append(f"Address: {self.error.address}")
        
        # Memory information
        if self.error.memory_info:
            sections.append(f"\nMemory Info: {self.error.memory_info}")
        
        # Stack trace
        if self.error.stack_trace:
            sections.append("\nStack Trace (Error Location):")
            for frame in self.error.stack_trace[:max_stack_depth]:
                sections.append(f"  {frame}")
            if len(self.error.stack_trace) > max_stack_depth:
                omitted = len(self.error.stack_trace) - max_stack_depth
                sections.append(f"  ... ({omitted} more frame{'s' if omitted > 1 else ''} omitted)")
        
        # Allocation trace
        if self.error.allocation_trace:
            sections.append("\nAllocation Stack Trace:")
            for frame in self.error.allocation_trace[:max_stack_depth]:
                sections.append(f"  {frame}")
            if len(self.error.allocation_trace) > max_stack_depth:
                omitted = len(self.error.allocation_trace) - max_stack_depth
                sections.append(f"  ... ({omitted} more frame{'s' if omitted > 1 else ''} omitted)")
        
        # Summary
        if self.error.summary:
            sections.append(f"\nSummary: {self.error.summary}")
        
        sections.append("=" * 60)
        
        result = '\n'.join(sections)
        
        # Optionally include raw output
        if include_raw:
            result += f"\n\nRaw ASAN Output:\n{self.raw_output}"
        
        return result
    
    def get_error_dict(self) -> Dict:
        """
        Get error information as a dictionary.
        
        Returns:
            Dictionary with all parsed error information
        """
        return {
            'error_type': self.error.error_type,
            'operation': self.error.operation,
            'size': self.error.size,
            'address': self.error.address,
            'file': self.error.file,
            'line': self.error.line,
            'column': self.error.column,
            'stack_trace': [
                {
                    'function': frame.function,
                    'file': frame.file,
                    'line': frame.line,
                    'column': frame.column
                }
                for frame in self.error.stack_trace
            ],
            'allocation_trace': [
                {
                    'function': frame.function,
                    'file': frame.file,
                    'line': frame.line,
                    'column': frame.column
                }
                for frame in self.error.allocation_trace
            ],
            'memory_info': self.error.memory_info,
            'summary': self.error.summary
        }
    
    def get_primary_location(self) -> Optional[Tuple[str, int]]:
        """
        Get the primary error location (file, line).
        
        Returns:
            Tuple of (file, line) or None if not available
        """
        if self.error.file and self.error.line:
            return (self.error.file, self.error.line)
        elif self.error.stack_trace and self.error.stack_trace[0].line:
            frame = self.error.stack_trace[0]
            return (frame.file, frame.line)
        return None


def parse_asan_for_llm(asan_output: str, **kwargs) -> str:
    """
    Convenience function to parse ASAN output and format for LLM.
    """
    parser = AsanParser(asan_output)
    return parser.format_for_llm(**kwargs)