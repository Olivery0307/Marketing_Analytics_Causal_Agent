Goal
Build a multi-agent system that performs the first three steps of a data analysis lifecycle: (1) collect data, (2) exploratory data analysis (3) form and communicate a hypothesis with evidence.

Like your first project, this agent will have a front-end, but its goal is to address the first half of a real data analyst workflow using real-world data.

 

The Three Steps
Every submission must implement all three steps.

Your agent doesn't need to run them in strict sequence, they can interleave, loop, and branch, this of this as a check-list. But all three must be present and clearly described in your README.

 

Step 1: Collect
Your agent retrieves real-world data from at least one external source. This cannot be hard-coded into the system prompt. The agent must actively go get the data at runtime.

Examples of collection methods (pick at least one):

SQL composition: The agent dynamically writes and executes SQL queries against a relational database or data file (SQLite, PostgreSQL, BigQuery, CSV/Avro/Parquet via DuckDB, etc.) based on the user's question.
API integration: The agent calls one or more public or private APIs (sports stats, weather, NYC Open Data, Census, etc.) and parses the structured responses.
RAG: The agent retrieves from an embedded document corpus (PDFs, epubs, markdown files embedded in a vector store) to ground its analysis in source material.
Web search / crawling: The agent uses a web search API or scrapes web pages at runtime to collect data.
What I'm Looking for: The data source has to be real, non-trivial, and relevant to the analytics question. A CSV with 50 rows that you hand-curated is not enough. The dataset must be large or abstract enough that it's not worth loading entirely into context.

Step 2: Explore and Analyze (EDA)
Your agent performs exploratory data analysis on the collected data. This means the agent does not jump straight to an answer. It examines the data first, then reasons about what it found.

The EDA phase must involve at least one tool call using some amount of the collected data.

Examples of qualifying EDA tool calls:

Statistical aggregation: A tool that computes means, medians, distributions, correlations, or growth rates over the retrieved data
Filtering and grouping: A tool that segments data by category, time period, or threshold and returns grouped results
Text analysis: An NLP tool that performs deterministic operations on text data such as sentiment counts, keyword extraction, entity frequency, topic clustering
Specialist sub-agent: A sub-agent with a dedicated analytical prompt, invoked via tool call or handoff (these count as your "at least one tool call"), that performs a focused analysis (e.g., a "trend detector" agent or a "comparison analyst" agent)
Code execution: The agent writes and executes Python (pandas, numpy, scipy, etc.) in a sandboxed environment to perform exploratory computation
These are just examples, there are other possibilities. If you have an idea and are unsure, please ask.

What I'm looking for: The EDA is dynamic. If a user asks a different question, the agent may use different tools or tools in different ways. The exploration surfaces something specific (a number, a pattern, an anomaly) that feeds into the hypothesis phase. A generic summary of the raw data is not EDA.

Step 3: Hypothesize
Your agent forms a hypothesis from the data and communicates it with supporting evidence. This is the analyst's deliverable, the "so what?" of the exploration.

Examples of hypothesis communication:

A natural language summary grounded in specific data points ("Revenue grew 12% in Q3, driven primarily by the Northeast region which saw a 23% increase")
A generated report or memo with tables and citations to the data
A visualization with an explanatory caption
A comparison of competing explanations with evidence for/against each
What we're looking for: The hypothesis is derived from the data, not the model weights. The agent explains its reasoning and which data points support the claim.

 

Core Requirements
Your agent must implement all required concepts and at least two concepts from the elective list.

In your README.md, identify which concepts you implemented and where in the codebase they live (file + function/class name).

Required:
Frontend: Your agent should have a front-end that I can load and interact with.
Agent framework: Your agent must use an agent framework (OpenAI Agents SDK, Google ADK, LangGraph, PydanticAI, CrewAI, etc.).
Tool calling: Your agent must use at least one tool call.
Non-trivial dataset: Your agent must retrieve data from a real, non-trivial external source at runtime. Bundled files are fine (CSV, Avro, SQLite, markdown, pdfs) but the dataset must be large enough that it can't be trivially dumped into context.
Multi-agent pattern: Your system must use at least one multi-agent pattern such as orchestrator-handoff, generator-critic, fan-out, agent-as-tool-call, etc. The important thing is that there are at least two distinct agents with different system prompts and responsibilities.
Deployed: Just like project 1, your application must be deployed and accessible.
README.md - A README.md explains how your project is run and how the three steps (Collect → EDA → Hypothesize) have been implemented.
 

Grab-Bag (at least two):
Your agent should also incorporate some advanced techniques.

Iterative refinement loop: The agent queries, analyzes, identifies gaps in its understanding, and queries again. Repeats until a stopping condition is met. This is the deep research pattern.
Code execution: The agent writes Python code (pandas, matplotlib, numpy, etc.) and executes it at runtime to transform data.
Artifacts: The agent writes persistent outputs to disk (CSVs, charts (PNG/SVG), markdown reports, or intermediate data tables).
Structured output: The agent uses structured output schemas (JSON mode, tool calling, or constrained decoding) to extract or emit reliable data structures at key points in the pipeline.
Second data retrieval method: Your agent uses a second, distinct data retrieval method from the list above (e.g., SQL + RAG, API + web crawling).
Data Visualization: Your agent generates and displays data visualizations to complement your hypothesis.
Parallel Execution: Your sub-agents run in parallel and the results awaited on and are aggregated into a final result.
 

Grading Breakdown
Step 1: Collect (5 pts) - Real data, retrieved at runtime, dynamic to different questions, accurately described in README
Step 2: Explore (5 pts) - At least one tool call that computes over collected data, surfaces specific findings, adapts to different questions
Step 3: Hypothesize (5 pts) - Grounded in data from previous steps, cites evidence, explains reasoning
Core Requirements (10 pts) - Frontend (2), framework (1), tool calling (1), non-trivial dataset (1), multi-agent pattern (2), deployed (2), README (1)
Grab Bag (5 pts) - 2.5 pts per elective concept, verified against code and README claims