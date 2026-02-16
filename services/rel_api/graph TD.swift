graph TD
    A["Document Input"] --> B["Document Type Detection"]
    B --> C{"Document Type?"}
    
    C -->|"Procedure"| D["Procedure Chunking (FIXED)"]
    C -->|"Technical"| E["Technical Chunking"]
    C -->|"General"| F["General Chunking"]
    
    D --> G["Major Section Boundary Detection"]
    D --> H["Minor Boundary Detection"]
    
    G --> I["SECTION 1: ENGINE OVERVIEW<br/>SECTION 2: MAINTENANCE<br/>Oil Change Procedure:"]
    H --> J["Step 1: Warm up engine<br/>Step 2: Turn off engine<br/>Problem: Engine won't start<br/>Every 5,000 miles:"]
    
    I --> K["Major boundaries<br/>start new chunks"]
    J --> L["Minor boundaries<br/>continue in same chunk<br/>unless too large"]
    
    K --> M["Well-Structured Chunks"]
    L --> M
    
    M --> N["✅ All procedure steps preserved<br/>✅ Section boundaries respected<br/>✅ Logical content grouping<br/>✅ Appropriate chunk sizes"]
    
    style A fill:#e1f5fe
    style N fill:#c8e6c9
    style D fill:#fff3e0
    style I fill:#f3e5f5
    style J fill:#f3e5f5