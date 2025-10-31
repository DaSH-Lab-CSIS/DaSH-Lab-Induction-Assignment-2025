# DaSH Lab Induction Assignment 2025
## Systems & GPU Track

Welcome to the DaSH Lab! This assignment is designed for students interested in **single-node systems and GPU computing**. Through this assignment, you'll gain hands-on experience with GPU programming and develop critical thinking skills by analyzing cutting-edge systems research.

---

## Table of Contents
1. [Introduction](#introduction)
2. [Assignment Overview](#assignment-overview)
3. [Part 1: GPU Matrix Multiplication](#part-1-gpu-matrix-multiplication)
4. [Part 2: Paper Analysis](#part-2-paper-analysis)
5. [Submission Guidelines](#submission-guidelines)
6. [Resources](#resources)
7. [Evaluation Criteria](#evaluation-criteria)

---

## Introduction

Modern computing systems increasingly rely on specialized hardware accelerators, particularly GPUs, to achieve high performance. Understanding how to effectively utilize these resources and critically analyze systems research are essential skills for any systems researcher.

This assignment will challenge you to:
- **Think like a performance engineer**: Optimize GPU code to approach library performance
- **Think like a researcher**: Critically analyze published research and identify strengths, weaknesses, and future directions

---

## Assignment Overview

**This assignment consists of TWO mandatory parts:**

### Part 1: GPU Matrix Multiplication (50%)
Implement and optimize matrix multiplication on GPU, comparing your performance against cuBLAS.

### Part 2: Research Paper Analysis (50%)
Read, understand, and critically analyze a systems research paper from our curated list.

---

## Part 1: GPU Matrix Multiplication

### Objective
Implement CUDA kernels for 1024×1024 matrix multiplication and optimize them to achieve performance as close as possible to cuBLAS, the highly optimized NVIDIA matrix multiplication library.

### Background
Matrix multiplication is a fundamental operation in scientific computing, machine learning, and graphics. While the algorithm is conceptually simple, achieving high performance on GPUs requires **deep understanding of how GPUs access memory**.

Key concepts you'll need to master:
- **GPU Memory Hierarchy**: Global memory (slow, large) → L2 cache → L1 cache → Shared memory (fast, small) → Registers (fastest, smallest)
- **Memory Access Patterns**: Understanding coalescing, alignment, bank conflicts, and cache behavior
- **Thread Organization**: How threads are organized into warps, blocks, and grids
- **Memory Bandwidth vs. Compute**: Is your kernel memory-bound or compute-bound?
- **Latency Hiding**: Overlapping memory transfers with computation
- **Hardware Limitations**: Peak memory bandwidth, compute throughput, occupancy

**The key challenge**: Moving data efficiently between memory levels while keeping compute units busy. Most naive implementations are severely memory-bound!

### Task Requirements

#### 1. Setup
- Use Google Colab for GPU access (can use personal computer if available): [Colab Reference Notebook](https://colab.research.google.com/drive/1zmP9yOGdMeG6Gjd7zkJ12FxtortfQJAi?usp=sharing)
- **Important**: 
  - Use your BITS email to access the notebook
  - This notebook is a **reference** to help you understand how to write and compile CUDA code in Colab
  - You should write your own code from scratch
- Ensure you have GPU runtime enabled (Runtime → Change runtime type → GPU)
- You will need to compile your `.cu` files using `nvcc` in Colab

#### 2. Implementation Requirements

Your goal is to implement **as many optimization techniques as possible** to get as close to cuBLAS performance as you can. 

**Approach this iteratively:**
1. Start with a simple naive implementation to understand the baseline
2. Profile it to identify bottlenecks
3. Apply one or two optimizations at a time
4. Measure the impact
5. Repeat until you approach cuBLAS performance or exhaust your optimization ideas

**Focus on understanding HOW GPUs ACCESS MEMORY**—this is the key to performance!

**Start with a Naive Baseline:**
- Basic global memory access
- One thread per output element
- Understand the baseline performance

**Then progressively optimize by thinking about how GPUs access memory:**


**You will be evaluated on:**
- How many distinct techniques you implement and combine
- Your understanding of WHY each technique helps (or doesn't)
- How close you get to cuBLAS performance
- Quality of your analysis and documentation

**Important**: Create separate `.cu` files for each major optimization milestone (e.g., `naive.cu`, `tiled.cu`, `tiled_coalesced.cu`, `advanced.cu`, etc.) so we can see your progression.

#### 3. Benchmarking Requirements

For each implementation:
- Measure execution time (average over multiple runs)
- Calculate GFLOPS (Giga Floating Point Operations Per Second)
- Compare against cuBLAS performance
- Report percentage of cuBLAS performance achieved

#### 4. Deliverables for Part 1

Create a folder `part1_matrix_multiplication/` containing:

**a) Code Files**
- `naive.cu` - Naive baseline implementation
- Multiple `.cu` files showing your optimization progression (e.g., `tiled.cu`, `tiled_coalesced.cu`, `optimized_v1.cu`, `optimized_v2.cu`, etc.)
- Name your files descriptively to indicate what optimizations they contain
- `benchmark.cu` - Benchmarking harness comparing ALL your versions + cuBLAS
- Each file should be independently compilable and runnable

**b) Report (`REPORT.md`)**

Your report should include:

1. **Implementation Details** (2-3 pages)
   - **Detailed description of EACH optimization technique used in every version**
   - Explain the theory behind each optimization (e.g., why shared memory helps, how tiling works)
   - Code snippets showing key optimizations
   - Compilation commands used (`nvcc` flags and options)
   - Why you chose these specific optimizations
   - Challenges encountered and how you solved them
   - What you tried that didn't work and why

2. **Performance Analysis** (2-3 pages)
   - Table comparing ALL your implementations:
     ```
     | Implementation             | Time (ms) | GFLOPS  | % of cuBLAS | Speedup vs Naive |
     |----------------------------|-----------|---------|-------------|------------------|
     | Naive                      |           |         |             | 1.0x             |
     | Tiled (shared memory)      |           |         |             |                  |
     | Tiled + Coalesced          |           |         |             |                  |
     | + Register blocking        |           |         |             |                  |
     | [Your optimizations...]    |           |         |             |                  |
     | cuBLAS                     |           |         |             |                  |
     ```
   - Performance graph showing progression (bar chart or line plot)
   - For each optimization step, explain:
     - What bottleneck you were trying to address
     - Expected vs. actual improvement
     - Why the improvement was (or wasn't) significant
   - Analysis of diminishing returns as you add more optimizations

3. **Profiling Results** (optional)
   - Use NVIDIA Nsight Compute or nvprof to profile your implementations
   - Report key metrics for at least your best 2-3 versions:
     - **Memory bandwidth utilization**: Are you saturating memory bandwidth?
     - **Compute utilization**: Are compute units fully utilized?
     - **Warp occupancy**: How many active warps per SM?
     - **Memory access patterns**: Are accesses coalesced? Any uncoalesced loads/stores?
     - **Cache hit rates**: L1/L2 cache performance
     - **Bank conflicts**: Shared memory bank conflicts detected?
   - Use profiling data to explain:
     - What is the bottleneck for each version? (Memory-bound vs. compute-bound)
     - How GPU memory hierarchy affects performance
     - Why certain optimizations had bigger impact than others

### Performance Targets

We understand cuBLAS is highly optimized. Here are rough targets:
- **Good**: Achieve >20% of cuBLAS performance with advanced optimizations
- **Excellent**: Achieve >40% of cuBLAS performance
- **Outstanding**: Achieve >60% of cuBLAS performance

*Note: Focus on the optimization journey and understanding, not just the final numbers.*

### Resources for Part 1
- [CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [Nsight Compute Documentation](https://docs.nvidia.com/nsight-compute/)
- [NVCC Compiler Documentation](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/)
- Reference Colab notebook (for understanding how to compile CUDA in Colab)

### Compilation Tips
- Basic compilation: `nvcc -o program program.cu`
- With optimization: `nvcc -O3 -o program program.cu`
- Linking cuBLAS: `nvcc -o program program.cu -lcublas`
- For profiling: Enable line-info with `-lineinfo` flag
- Check GPU architecture: Use `-arch=sm_XX` (check your Colab GPU model)

---

## Part 2: Paper Analysis

### Objective
Demonstrate your ability to critically read, understand, and analyze systems research by selecting and critiquing one paper from our curated list.

### Paper Selection

Choose **ONE** paper from the following list:

#### 1. **The Linux Scheduler: A Decade of Wasted Cores**
- **Conference**: EuroSys 2016
- **Link**: https://people.ece.ubc.ca/sasha/papers/eurosys16-final29.pdf
- **Focus**: Critical analysis of Linux scheduler identifying four performance bugs causing significant core underutilization
- **Why Read**: Understanding real-world scheduler design challenges in production systems
- **Topics**: Scheduling, load balancing, multicore systems, Linux kernel

#### 2. **Efficient Memory Management for Large Language Model Serving with PagedAttention**
- **Conference**: SOSP 2023
- **Link**: https://arxiv.org/abs/2309.06180
- **Focus**: Applies classical virtual memory and paging techniques from operating systems to LLM serving (vLLM system)
- **Why Read**: Demonstrates how OS principles apply to modern ML systems; highly relevant to current AI infrastructure
- **Topics**: Memory management, virtual memory, LLM serving, GPU memory optimization

#### 3. **SILT: A Memory-Efficient, High-Performance Key-Value Store**
- **Conference**: SOSP 2011
- **Link**: https://dl.acm.org/doi/10.1145/2043556.2043558
- **Focus**: Memory-efficient key-value store based on flash storage that scales to billions of items
- **Why Read**: Foundational work on flash-based storage systems; principles still relevant today
- **Topics**: Key-value stores, flash storage, memory efficiency, data structures

#### 4. **Optimizing Memory-mapped I/O for Fast Storage Devices**
- **Conference**: USENIX ATC 2020
- **Link**: https://www.usenix.org/system/files/atc20-papagiannis.pdf
- **Focus**: Memory-mapped I/O optimization for low-latency storage devices
- **Why Read**: Addresses mismatch between traditional I/O interfaces and modern fast storage
- **Topics**: Memory-mapped I/O, NVMe, storage performance, OS I/O stack

### Analysis Requirements

Create a folder `part2_paper_analysis/` containing a comprehensive analysis document (`ANALYSIS.md`) with the following sections:

#### 1. **Summary** (1 page)
- What problem does the paper address?
- Why is this problem important?
- What is the key insight or main contribution?
- Brief overview of the proposed solution

#### 2. **Technical Understanding** (2-3 pages)

**a) Problem Analysis**
- Detailed explanation of the problem
- Why existing solutions are inadequate
- Motivating examples or workloads

**b) Proposed Solution**
- System design and architecture
- Key algorithms or techniques
- Implementation details (where relevant)

**c) Evaluation**
- Experimental setup
- Key results and metrics
- How results support the claims

#### 3. **Critical Analysis** (2-3 pages)

**Strengths**
- What does the paper do well?
- Novel insights or techniques
- Strong experimental evidence
- Clear presentation (or not)

**Weaknesses**
- What are the limitations?
- Questionable assumptions
- Missing experiments or evaluations
- Narrow scope or applicability

**Specific Critiques** (choose at least 3):
- Are the evaluation metrics appropriate?
- Is the baseline comparison fair?
- Are there overlooked edge cases?
- Does the solution generalize beyond the tested scenarios?
- Are there hidden costs not discussed?
- Is the system practical to deploy?
- Are the claimed contributions novel?

#### 4. **Personal Reflection** (1 page)

- What did you learn?
- What surprised you?
- How does this relate to other systems concepts you know?
- Would you have designed it differently? How?

### Analysis Guidelines

**Do:**
- Be specific with examples from the paper
- Support criticisms with technical reasoning
- Consider both theoretical and practical aspects
- Think about real-world deployment challenges
- Connect ideas to broader systems concepts

**Don't:**
- Just summarize without analysis
- Make vague criticisms ("not scalable" without explanation)
- Focus only on writing quality
- Ignore experimental methodology
- Accept claims without scrutiny

### Evaluation Criteria for Part 2

- **Understanding** (30%): Demonstrates deep comprehension of technical content
- **Critical Thinking** (40%): Provides insightful, well-reasoned critiques
- **Breadth** (15%): Considers multiple perspectives and broader context
- **Clarity** (15%): Well-organized, clear writing with proper citations

---

## Submission Guidelines

### File Structure
```
DaSH-Lab-Induction-Assignment-2025/
├── part1_matrix_multiplication/
│   ├── naive.cu                    # Naive baseline implementation
│   ├── tiled.cu                    # Shared memory tiling
│   ├── tiled_coalesced.cu          # Tiling + memory coalescing
│   ├── optimized_v1.cu             # Multiple techniques combined
│   ├── optimized_v2.cu             # Advanced optimizations
│   ├── [more .cu files...]         # Add as many optimization versions as you explore
│   ├── benchmark.cu                # Benchmarking ALL versions + cuBLAS
│   ├── Makefile (or COMPILE.md)    # Compilation instructions for all files
│   ├── REPORT.md                   # Detailed writeup with all results
│   └── notebook.ipynb              # Complete Colab notebook with all implementations
├── part2_paper_analysis/
│   ├── ANALYSIS.md                 # Critical analysis writeup
│   └── paper.pdf                   # Copy of the paper you analyzed
└── README.md                       # Project overview (this file)
```

### Submission Instructions

1. **GitHub Repository (MANDATORY)**
   - Clone this repository or create a new **private** repository
   - Your repository MUST contain:
     - **ALL source code** (`.cu` files, compilation scripts)
     - **ALL reports** (`REPORT.md` for Part 1, `ANALYSIS.md` for Part 2)
     - **Colab notebook** (`.ipynb` file)
     - **Compilation instructions** (Makefile or COMPILE.md)
     - A clear README explaining how to run your code
   - Ensure all code is **well-commented** with explanations of techniques
   - Code must **compile and run successfully**
   - Do NOT submit code that doesn't compile—fix all errors before submission
   - **Keep your development history**: We want to see your progress through commits

2. **Deadline**: See the main branch

3. **Submission Format**:
   - Email repository link to: see the main branch for submission guidelines

### Academic Integrity

- You may discuss concepts with peers, but **all code and writing must be your own**
- Cite any external resources, tutorials, or papers you reference
- Do not copy-paste code from GitHub/StackOverflow without understanding and attribution
- Using AI assistants (ChatGPT, Copilot) is allowed for learning, but you must understand and be able to explain all submitted code

---

## Resources

### CUDA Programming
- [CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [Nsight Compute](https://docs.nvidia.com/nsight-compute/)
- [NVIDIA CUDA Samples](https://github.com/NVIDIA/cuda-samples)

### Systems Research
- [How to Read a Paper (Keshav)](http://ccr.sigcomm.org/online/files/p83-keshavA.pdf)
- [Gernot's List of Systems Benchmarking Crimes (Advanced)](https://gernot-heiser.org/benchmarking-crimes.html)
---

## Evaluation Criteria

### Part 1: GPU Matrix Multiplication (50 points)

| Criterion | Points | Description |
|-----------|--------|-------------|
| Correctness | 10 | All implementations produce correct results |
| Implementation Quality | 10 | Clean code, proper memory management, good practices |
| Optimization Breadth & Depth | 14 | Number and variety of optimization techniques explored; understanding of GPU memory access patterns |
| Performance | 8 | Achieved performance relative to cuBLAS and improvement over baseline |
| Report Quality | 8 | Clear explanation of each technique, insightful analysis, profiling data, documentation of failures |

### Part 2: Paper Analysis (50 points)

| Criterion | Points | Description |
|-----------|--------|-------------|
| Technical Understanding | 16 | Demonstrates comprehension of paper's technical content |
| Critical Analysis | 18 | Insightful critiques with technical reasoning |
| Breadth | 8 | Considers broader context and future directions |
| Clarity & Organization | 8 | Well-structured, clear writing, proper citations |

### Total: 100 points

**Minimum Passing Score**: 60/100

### What Makes a Strong Submission?

**For Part 1 (GPU Matrix Multiplication):**
- ✅ 5+ different optimization versions showing clear progression
- ✅ Deep understanding of GPU memory access patterns demonstrated in code and writeup
- ✅ Comprehensive profiling data explaining bottlenecks
- ✅ Documentation of failed attempts with explanations
- ✅ All code compiles and runs successfully
- ✅ Achieves >30% of cuBLAS performance (or shows strong optimization effort if lower)
- ✅ Clear explanations of why each technique helps or hurts performance

**For Part 2 (Paper Analysis):**
- ✅ Demonstrates understanding beyond surface-level summary
- ✅ Specific technical critiques with reasoning
- ✅ Connects paper to broader systems concepts
- ✅ Thoughtful discussion of limitations and future work
- ✅ Well-structured writing with proper citations

#### **This will be followed by rigrous interview to select candidates**
---

### Common Questions

**Q: Can I use libraries other than cuBLAS for comparison?**  
A: Yes, but cuBLAS must be your primary baseline. You can additionally compare against other libraries.

**Q: What if I can't achieve good performance?**  
A: Focus on the learning process. Document what you tried, what worked, what didn't, and why. This is more important than raw numbers. Show us you understand GPU memory access patterns and made serious optimization attempts.

**Q: How many different implementations should I submit?**  
A: There's no fixed number. Start with naive and keep optimizing until you're close to cuBLAS or have exhausted techniques you can implement. Students who try 5-7+ different optimization approaches typically demonstrate better understanding than those who stop at 3.

**Q: Can I analyze a paper not on the list?**  
A: Please ask first. We've curated this list to cover diverse systems topics at appropriate difficulty levels.

**Q: How long should the reports be?**  
A: Follow the page guidelines for each section. Quality over quantity—be concise but thorough.

**Q: Can I work in teams?**  
A: No, this is an individual assessment. However, you may discuss concepts with peers.

**Q: Do I need to keep all my code attempts, even failed ones?**  
A: Yes! Document your optimization journey. Include comments in your code or report explaining what didn't work and why. This shows your learning process.

**Q: My code doesn't compile. Can I still submit?**  
A: No. All submitted code MUST compile and run successfully. Debug and fix all compilation errors before submission.

**Q: Should I include the Colab notebook AND standalone .cu files?**  
A: Yes, both. The .cu files should be standalone and compilable, and the notebook should demonstrate the complete workflow including compilation.

---

## Welcome to DaSH Lab!

We're excited to see your work! This assignment is challenging, but it will give you a solid foundation in both practical GPU programming and critical research analysis—essential skills for systems research.

Remember: The goal is learning, not perfection. Show us your thought process, your struggles, and your insights.

Good luck!

---

**Last Updated**: October 31, 2025  
**Version**: 1.1

