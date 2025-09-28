#!/usr/bin/env python3
"""
SQL Injection debug script
Test hangi pattern'in 'ingilizce seviyesi: b1.' girdisini yakaladığını bulur
"""

import re

# SQL Injection patterns from security_utils.py (UPDATED)
SQL_INJECTION_PATTERNS = [
    r"(\b(union|select|insert|update|delete|drop|create|alter|exec|execute)\b)",
    r"(--|#|/\*|\*/)",
    r"(\bor\b\s+\d+\s*=\s*\d+)",
    r"(\band\b\s+\d+\s*=\s*\d+)",
    r"('|\")(\s)*(or|and|union|select)(\s)+(\w+\s*=|\w+\s*\(|from\s+\w+|into\s+\w+|\d+)",  # More specific SQL context
    r"\b(script|javascript|vbscript|onload|onerror|onclick)\b",
    r"(\<\s*script)",
    r"(\bxp_cmdshell\b)"
]

def debug_sql_injection(text: str):
    """Test hangi pattern'in bu metni yakaladığını bulur"""
    try:
        print(f"Testing text: '{text[:100]}...'")
    except UnicodeEncodeError:
        print(f"Testing text: [Text with special characters, length: {len(text)}]")
    print("=" * 50)

    # Compile regex patterns for performance
    sql_patterns = [re.compile(p, re.IGNORECASE) for p in SQL_INJECTION_PATTERNS]

    for i, pattern in enumerate(sql_patterns):
        match = pattern.search(text)
        if match:
            print(f"[X] MATCH FOUND!")
            print(f"Pattern {i}: {SQL_INJECTION_PATTERNS[i]}")
            print(f"Compiled pattern: {pattern.pattern}")
            try:
                print(f"Match: {match.group()}")
                print(f"Match position: {match.start()}-{match.end()}")
                print(f"Match groups: {match.groups()}")
            except UnicodeEncodeError:
                print(f"Match found but contains special characters")
                print(f"Match position: {match.start()}-{match.end()}")
        else:
            print(f"[OK] Pattern {i}: NO MATCH - {SQL_INJECTION_PATTERNS[i]}")

    print("=" * 50)

if __name__ == "__main__":
    # Test the ACTUAL problematic input from Android app
    actual_android_input = """ingilizce seviyesi: b1.

Conversation history:
Teacher: Hello! I'm your English learning AI assistant! 🌟

I can help you improve your English at different levels (A1-C2). You can:
• Practice conversations
• Learn grammar rules
• Expand your vocabulary
• Get writing help

To get started, you can specify your level like: "ingilizce seviyesi: b1" or just start chatting!

What would you like to practice today?
Student: Help me with grammar

New message: Help me with grammar"""

    print("Testing the ACTUAL Android app input:")
    debug_sql_injection(actual_android_input)

    print("\n" + "=" * 60)
    print("Testing just the simple input:")
    simple_input = "ingilizce seviyesi: b1."
    debug_sql_injection(simple_input)

    print("\nTesting other inputs:")
    print("-" * 30)

    # Test some legitimate inputs and SQL injections
    test_inputs = [
        "Hello, how are you?",
        "ingilizce seviyesi: a2.",
        "Please help me with English",
        'You can do "this" or "that" option',  # Should NOT trigger (legitimate English)
        "select * from users",  # Real SQL injection - should trigger
        "union select 1,2,3",   # Real SQL injection - should trigger
        "Teacher: Hello! I'm your English learning AI assistant",
        # Test the specific new pattern requirements:
        "' or 1=1",           # Should trigger (SQL injection)
        "' or user_id=1",     # Should trigger (SQL injection)
        "' union select",     # Should NOT trigger (incomplete)
        "' or select from users",  # Should trigger (SQL injection)
        "\" and password=admin",   # Should trigger (SQL injection)
        "You can \"read\" or \"write\" files",  # Should NOT trigger (legitimate English)
    ]

    for test_input in test_inputs:
        print(f"\nTesting: '{test_input}'")
        debug_sql_injection(test_input)