#!/usr/bin/env python3
"""
Phase 2: File-Based Code Analysis System
Real code interaction without Claude CLI, demonstrating concrete analysis.
"""

import ast
import os
import re
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime


@dataclass
class CodePattern:
    """Represents a discovered code pattern."""
    pattern_type: str
    location: str
    description: str
    severity: str  # low, medium, high
    fix_complexity: int  # 1-10
    estimated_tokens: int


@dataclass
class RefactoringProposal:
    """Concrete refactoring proposal with before/after code."""
    file_path: str
    line_start: int
    line_end: int
    pattern: CodePattern
    original_code: str
    proposed_code: str
    explanation: str
    confidence: float


class FileBasedCodeAnalyzer:
    """
    Analyzes code files and generates concrete improvement proposals.
    No mocks - real AST analysis and pattern detection.
    """
    
    def __init__(self, worktree_path: str):
        self.worktree_path = Path(worktree_path)
        self.patterns_found = []
        self.proposals = []
        
    def analyze_repository(self, max_files: int = 10) -> Dict[str, Any]:
        """Analyze Python files in the repository."""
        results = {
            'files_analyzed': 0,
            'patterns_found': 0,
            'proposals_generated': 0,
            'total_estimated_tokens': 0,
            'file_results': []
        }
        
        # Find Python files
        python_files = list(self.worktree_path.rglob("*.py"))[:max_files]
        
        for py_file in python_files:
            if self._should_skip_file(py_file):
                continue
                
            file_result = self.analyze_file(py_file)
            results['file_results'].append(file_result)
            results['files_analyzed'] += 1
            results['patterns_found'] += len(file_result['patterns'])
            results['proposals_generated'] += len(file_result['proposals'])
            results['total_estimated_tokens'] += file_result['estimated_tokens']
            
        return results
    
    def analyze_file(self, file_path: Path) -> Dict[str, Any]:
        """Perform deep analysis on a single file."""
        result = {
            'file': str(file_path.relative_to(self.worktree_path)),
            'patterns': [],
            'proposals': [],
            'metrics': {},
            'estimated_tokens': 0
        }
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.splitlines()
                
            # Basic metrics
            result['metrics'] = {
                'lines': len(lines),
                'characters': len(content),
                'functions': 0,
                'classes': 0
            }
            
            # Pattern detection
            patterns = []
            
            # 1. TODO/FIXME detection with context
            patterns.extend(self._find_todo_patterns(content, lines))
            
            # 2. AST-based analysis
            try:
                tree = ast.parse(content)
                patterns.extend(self._analyze_ast_patterns(tree, lines))
                
                # Count functions and classes
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        result['metrics']['functions'] += 1
                    elif isinstance(node, ast.ClassDef):
                        result['metrics']['classes'] += 1
                        
            except SyntaxError:
                patterns.append(CodePattern(
                    pattern_type='syntax_error',
                    location=str(file_path),
                    description='File has syntax errors',
                    severity='high',
                    fix_complexity=8,
                    estimated_tokens=1000
                ))
            
            # 3. Code smell detection
            patterns.extend(self._detect_code_smells(content, lines))
            
            # Store patterns
            result['patterns'] = patterns
            self.patterns_found.extend(patterns)
            
            # Generate refactoring proposals
            for pattern in patterns:
                proposal = self._generate_proposal(pattern, file_path, lines)
                if proposal:
                    result['proposals'].append(proposal)
                    self.proposals.append(proposal)
                    result['estimated_tokens'] += proposal.pattern.estimated_tokens
                    
        except Exception as e:
            result['error'] = str(e)
            
        return result
    
    def _should_skip_file(self, file_path: Path) -> bool:
        """Check if file should be skipped."""
        skip_patterns = ['__pycache__', '.git', 'venv', 'env', '.tox']
        return any(pattern in str(file_path) for pattern in skip_patterns)
    
    def _find_todo_patterns(self, content: str, lines: List[str]) -> List[CodePattern]:
        """Find TODO/FIXME comments with context."""
        patterns = []
        todo_regex = re.compile(r'#\s*(TODO|FIXME|HACK|XXX|BUG):\s*(.+)', re.IGNORECASE)
        
        for i, line in enumerate(lines):
            match = todo_regex.search(line)
            if match:
                todo_type = match.group(1).upper()
                todo_text = match.group(2).strip()
                
                # Determine complexity based on context
                complexity = 5  # default
                if 'implement' in todo_text.lower():
                    complexity = 7
                elif 'refactor' in todo_text.lower():
                    complexity = 6
                elif 'fix' in todo_text.lower():
                    complexity = 4
                    
                patterns.append(CodePattern(
                    pattern_type=f'todo_{todo_type.lower()}',
                    location=f"line {i+1}",
                    description=f"{todo_type}: {todo_text}",
                    severity='medium',
                    fix_complexity=complexity,
                    estimated_tokens=complexity * 100
                ))
                
        return patterns
    
    def _analyze_ast_patterns(self, tree: ast.AST, lines: List[str]) -> List[CodePattern]:
        """Analyze AST for code patterns."""
        patterns = []
        
        # Check for various patterns
        for node in ast.walk(tree):
            # Long functions
            if isinstance(node, ast.FunctionDef):
                if hasattr(node, 'lineno') and hasattr(node, 'end_lineno'):
                    func_lines = node.end_lineno - node.lineno
                    if func_lines > 50:
                        patterns.append(CodePattern(
                            pattern_type='long_function',
                            location=f"{node.name} (lines {node.lineno}-{node.end_lineno})",
                            description=f"Function '{node.name}' is {func_lines} lines long",
                            severity='medium',
                            fix_complexity=7,
                            estimated_tokens=func_lines * 10
                        ))
                        
                # Check for missing docstrings
                if not ast.get_docstring(node):
                    patterns.append(CodePattern(
                        pattern_type='missing_docstring',
                        location=f"{node.name} (line {node.lineno})",
                        description=f"Function '{node.name}' lacks documentation",
                        severity='low',
                        fix_complexity=3,
                        estimated_tokens=200
                    ))
                    
                # Complex functions (high cyclomatic complexity)
                complexity = self._calculate_complexity(node)
                if complexity > 10:
                    patterns.append(CodePattern(
                        pattern_type='high_complexity',
                        location=f"{node.name} (line {node.lineno})",
                        description=f"Function '{node.name}' has complexity {complexity}",
                        severity='high',
                        fix_complexity=8,
                        estimated_tokens=complexity * 50
                    ))
                    
            # Unused imports
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if not self._is_name_used(alias.name, tree):
                        patterns.append(CodePattern(
                            pattern_type='unused_import',
                            location=f"line {node.lineno}",
                            description=f"Unused import: {alias.name}",
                            severity='low',
                            fix_complexity=1,
                            estimated_tokens=50
                        ))
                        
        return patterns
    
    def _calculate_complexity(self, node: ast.FunctionDef) -> int:
        """Calculate cyclomatic complexity of a function."""
        complexity = 1
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
                
        return complexity
    
    def _is_name_used(self, name: str, tree: ast.AST) -> bool:
        """Check if an imported name is used in the code."""
        base_name = name.split('.')[0]
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id == base_name:
                if not isinstance(node.ctx, ast.Store):
                    return True
            elif isinstance(node, ast.Attribute) and node.attr == base_name:
                return True
                
        return False
    
    def _detect_code_smells(self, content: str, lines: List[str]) -> List[CodePattern]:
        """Detect common code smells."""
        patterns = []
        
        # Magic numbers
        magic_number_regex = re.compile(r'(?<!["\w])(\d{2,})(?!["\w])')
        for i, line in enumerate(lines):
            # Skip comments and strings
            if '#' in line:
                line = line[:line.index('#')]
            
            matches = magic_number_regex.findall(line)
            for match in matches:
                if int(match) not in [10, 100, 1000, 60, 24, 365]:  # Common non-magic numbers
                    patterns.append(CodePattern(
                        pattern_type='magic_number',
                        location=f"line {i+1}",
                        description=f"Magic number {match} should be a named constant",
                        severity='low',
                        fix_complexity=2,
                        estimated_tokens=100
                    ))
                    break  # One per line
                    
        # Duplicate code detection (simplified)
        # Look for identical non-trivial lines
        line_counts = {}
        for i, line in enumerate(lines):
            stripped = line.strip()
            if len(stripped) > 20 and not stripped.startswith('#'):
                if stripped in line_counts:
                    line_counts[stripped].append(i)
                else:
                    line_counts[stripped] = [i]
                    
        for line_text, occurrences in line_counts.items():
            if len(occurrences) > 2:
                patterns.append(CodePattern(
                    pattern_type='duplicate_code',
                    location=f"lines {', '.join(str(l+1) for l in occurrences[:3])}...",
                    description=f"Duplicate code: '{line_text[:50]}...'",
                    severity='medium',
                    fix_complexity=5,
                    estimated_tokens=300
                ))
                
        return patterns
    
    def _generate_proposal(self, pattern: CodePattern, file_path: Path, 
                          lines: List[str]) -> Optional[RefactoringProposal]:
        """Generate concrete refactoring proposal for a pattern."""
        
        # Extract line number from location
        line_match = re.search(r'line (\d+)', pattern.location)
        if not line_match:
            return None
            
        line_num = int(line_match.group(1)) - 1
        if line_num >= len(lines):
            return None
            
        # Generate proposals based on pattern type
        if pattern.pattern_type == 'unused_import':
            return self._propose_remove_import(pattern, file_path, lines, line_num)
        elif pattern.pattern_type == 'magic_number':
            return self._propose_extract_constant(pattern, file_path, lines, line_num)
        elif pattern.pattern_type.startswith('todo_'):
            return self._propose_implement_todo(pattern, file_path, lines, line_num)
        elif pattern.pattern_type == 'missing_docstring':
            return self._propose_add_docstring(pattern, file_path, lines, line_num)
            
        return None
    
    def _propose_remove_import(self, pattern: CodePattern, file_path: Path, 
                              lines: List[str], line_num: int) -> RefactoringProposal:
        """Propose removing unused import."""
        original = lines[line_num]
        
        return RefactoringProposal(
            file_path=str(file_path),
            line_start=line_num + 1,
            line_end=line_num + 1,
            pattern=pattern,
            original_code=original,
            proposed_code="",  # Remove the line
            explanation="Remove unused import to clean up dependencies",
            confidence=0.95
        )
    
    def _propose_extract_constant(self, pattern: CodePattern, file_path: Path,
                                 lines: List[str], line_num: int) -> RefactoringProposal:
        """Propose extracting magic number to constant."""
        original = lines[line_num]
        
        # Extract the magic number
        match = re.search(r'(\d{2,})', original)
        if not match:
            return None
            
        magic_num = match.group(1)
        
        # Generate constant name based on context
        const_name = f"DEFAULT_VALUE_{magic_num}"
        if 'timeout' in original.lower():
            const_name = f"TIMEOUT_SECONDS"
        elif 'max' in original.lower():
            const_name = f"MAX_VALUE"
        elif 'min' in original.lower():
            const_name = f"MIN_VALUE"
            
        # Create proposal with constant definition
        proposed = f"{const_name} = {magic_num}\n# ... (at module level)\n{original.replace(magic_num, const_name)}"
        
        return RefactoringProposal(
            file_path=str(file_path),
            line_start=line_num + 1,
            line_end=line_num + 1,
            pattern=pattern,
            original_code=original,
            proposed_code=proposed,
            explanation=f"Extract magic number {magic_num} to named constant {const_name}",
            confidence=0.8
        )
    
    def _propose_implement_todo(self, pattern: CodePattern, file_path: Path,
                               lines: List[str], line_num: int) -> RefactoringProposal:
        """Propose implementation for TODO."""
        original = lines[line_num]
        todo_text = pattern.description
        
        # Simple implementation suggestions based on TODO content
        if 'memoization' in todo_text.lower():
            proposed = '''    # TODO: Add memoization for better performance
    @functools.lru_cache(maxsize=128)
    def memoized_function(...):
        # Implementation here'''
        elif 'validation' in todo_text.lower():
            proposed = '''    # TODO: Add input validation
    if not isinstance(input_value, expected_type):
        raise ValueError(f"Expected {expected_type}, got {type(input_value)}")
    if input_value < 0:
        raise ValueError("Value must be non-negative")'''
        elif 'error handling' in todo_text.lower():
            proposed = '''    # TODO: Add error handling
    try:
        # Existing code here
        result = risky_operation()
    except SpecificError as e:
        logger.error(f"Operation failed: {e}")
        raise
    except Exception as e:
        logger.exception("Unexpected error")
        raise RuntimeError("Operation failed") from e'''
        else:
            # Generic implementation
            proposed = f"    # {original.strip()}\n    # Implementation needed here\n    raise NotImplementedError('{todo_text}')"
            
        return RefactoringProposal(
            file_path=str(file_path),
            line_start=line_num + 1,
            line_end=line_num + 1,
            pattern=pattern,
            original_code=original,
            proposed_code=proposed,
            explanation=f"Implement {pattern.description}",
            confidence=0.6
        )
    
    def _propose_add_docstring(self, pattern: CodePattern, file_path: Path,
                              lines: List[str], line_num: int) -> RefactoringProposal:
        """Propose adding docstring to function."""
        # Find the function definition
        func_line = None
        for i in range(max(0, line_num - 5), min(len(lines), line_num + 5)):
            if 'def ' in lines[i] and '(' in lines[i]:
                func_line = i
                break
                
        if func_line is None:
            return None
            
        original = lines[func_line]
        func_match = re.search(r'def\s+(\w+)\s*\((.*?)\)', original)
        if not func_match:
            return None
            
        func_name = func_match.group(1)
        params = func_match.group(2)
        
        # Generate docstring
        indent = len(original) - len(original.lstrip())
        docstring_lines = [
            original.rstrip(),
            ' ' * (indent + 4) + '"""',
            ' ' * (indent + 4) + f'{func_name.replace("_", " ").title()} function.',
            ' ' * (indent + 4) + '',
        ]
        
        # Add parameter documentation
        if params.strip():
            docstring_lines.append(' ' * (indent + 4) + 'Args:')
            for param in params.split(','):
                param_name = param.strip().split(':')[0].split('=')[0].strip()
                if param_name and param_name not in ['self', 'cls']:
                    docstring_lines.append(' ' * (indent + 8) + f'{param_name}: Description here')
                    
        docstring_lines.extend([
            ' ' * (indent + 4) + '',
            ' ' * (indent + 4) + 'Returns:',
            ' ' * (indent + 8) + 'Description of return value',
            ' ' * (indent + 4) + '"""'
        ])
        
        proposed = '\n'.join(docstring_lines)
        
        return RefactoringProposal(
            file_path=str(file_path),
            line_start=func_line + 1,
            line_end=func_line + 1,
            pattern=pattern,
            original_code=original,
            proposed_code=proposed,
            explanation=f"Add docstring to function '{func_name}'",
            confidence=0.9
        )
    
    def get_summary_report(self) -> Dict[str, Any]:
        """Generate summary report of analysis."""
        pattern_counts = {}
        for pattern in self.patterns_found:
            pattern_counts[pattern.pattern_type] = pattern_counts.get(pattern.pattern_type, 0) + 1
            
        return {
            'total_patterns': len(self.patterns_found),
            'total_proposals': len(self.proposals),
            'pattern_breakdown': pattern_counts,
            'estimated_total_tokens': sum(p.pattern.estimated_tokens for p in self.proposals),
            'high_severity_count': sum(1 for p in self.patterns_found if p.severity == 'high'),
            'medium_severity_count': sum(1 for p in self.patterns_found if p.severity == 'medium'),
            'low_severity_count': sum(1 for p in self.patterns_found if p.severity == 'low')
        }