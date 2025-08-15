# GPU Examples for HPC Clusters

This directory contains GPU benchmarking and demonstration scripts designed to showcase H100 GPU capabilities for high-performance computing clusters. These examples are specifically tailored for **IT Support Staff**, **Cluster Administrators**, and **Social Science Researchers**.

## 🎯 **Target Audience**

### **IT Support & Cluster Administrators**
- **Purpose**: Validate GPU hardware performance and cluster configuration
- **Focus**: System performance metrics, resource utilization, and scalability
- **Benefits**: Identify bottlenecks, optimize cluster settings, and demonstrate ROI

### **Social Science Researchers**
- **Purpose**: Showcase how GPU computing accelerates research workflows
- **Focus**: Real-world applications in economics, psychology, sociology, and political science
- **Benefits**: Understand GPU advantages for large-scale data analysis and simulations

## 🚀 **H100 Interactive Demo**

### **What is the H100 Demo?**
The H100 demo is an interactive showcase that runs comprehensive performance tests on NVIDIA H100 GPUs, comparing them against traditional CPU-based computing.

### **How to Run**
```bash
# From a login node, launch the interactive demo
./h100_demo.sh
```

### **What Happens During the Demo**
1. **Launches interactive SLURM session** on the H100 partition
2. **Runs GPU performance tests** (Phases 1-5)
3. **Runs CPU comparison tests** (Phases 6-9)
4. **Provides comprehensive summary** with speedup calculations
5. **Optional real-time monitoring** of GPU utilization

## 📊 **Detailed Simulation Explanations**

### **Phase 1: Matrix Operations**
**What it does**: Multiplies large matrices together (like giant spreadsheets)

**Real-world analogy**: 
- Imagine you have two massive Excel spreadsheets with 32,768 rows and columns
- You need to multiply every number in the first spreadsheet with every number in the second
- This creates over 1 billion calculations

**Why it matters**:
- **Economics**: Portfolio optimization, risk modeling
- **Psychology**: Factor analysis, structural equation modeling
- **Sociology**: Network analysis, social influence calculations
- **IT Support**: Tests raw computational power and memory bandwidth

**Expected results**: H100 processes these massive matrices 50-100x faster than CPU

**Learn more**:
- [Matrix Multiplication Explained](https://www.khanacademy.org/math/precalculus/x9e81a4f98389efdf:matrices/x9e81a4f98389efdf:multiplying-matrices-by-matrices/v/matrix-multiplication-intro)
- [GPU Matrix Operations](https://developer.nvidia.com/blog/cublas-strided-batched-matrix-operations/)
- [PyTorch Matrix Operations](https://pytorch.org/docs/stable/generated/torch.mm.html)

### **Phase 2: Memory Bandwidth**
**What it does**: Tests how fast data can move between CPU memory and GPU memory

**Real-world analogy**:
- Think of it like testing how fast you can transfer files between your computer's hard drive and a USB drive
- The GPU has its own high-speed memory (like a super-fast USB drive)
- We test how quickly we can move large amounts of data back and forth

**Why it matters**:
- **Researchers**: Large datasets need to be transferred to GPU for processing
- **IT Support**: Identifies memory bottlenecks and transfer limitations
- **Cluster Admins**: Helps optimize data pipeline configurations

**Expected results**: H100 can transfer 16GB of data in seconds at 3-4 TB/s speeds

**Learn more**:
- [GPU Memory Architecture](https://developer.nvidia.com/blog/maximizing-unified-memory-performance-cuda/)
- [CUDA Memory Management](https://developer.nvidia.com/blog/unified-memory-cuda-beginners/)
- [PyTorch GPU Memory](https://pytorch.org/docs/stable/notes/cuda.html#memory-management)

### **Phase 3: Parallel Processing**
**What it does**: Performs mathematical operations on millions of data points simultaneously

**Real-world analogy**:
- Imagine you have 100 million numbers and need to calculate the sine, cosine, and square root of each
- CPU does this one number at a time (like a single worker)
- GPU does this on all 100 million numbers at once (like 10,000 workers)

**Why it matters**:
- **Psychology**: Processing large survey datasets
- **Sociology**: Analyzing social network connections
- **Economics**: Monte Carlo simulations for risk assessment
- **IT Support**: Demonstrates parallel processing capabilities

**Expected results**: H100 processes 100 million elements 50-100x faster than CPU

**Learn more**:
- [Parallel Computing Basics](https://www.nvidia.com/en-us/data-center/gpu-accelerated-applications/parallel-computing/)
- [CUDA Parallel Programming](https://developer.nvidia.com/blog/even-easier-introduction-cuda/)
- [PyTorch Parallel Operations](https://pytorch.org/docs/stable/notes/cuda.html#asynchronous-execution)

### **Phase 4: Monte Carlo Econometrics**
**What it does**: Simulates thousands of possible financial scenarios to assess risk

**Real-world analogy**:
- Imagine you're a financial advisor trying to predict portfolio performance
- Instead of guessing, you run 75,000 different scenarios with different market conditions
- This gives you a realistic picture of potential gains and losses

**Specific simulations**:

#### **Asset Pricing Simulation**
- **What**: Simulates stock prices over 252 trading days (1 year)
- **Scale**: 75,000 different scenarios with 100 different stocks
- **Output**: Value-at-Risk (VaR) and expected returns
- **Real-world use**: Portfolio risk management, investment strategy

#### **Bootstrap Confidence Intervals**
- **What**: Repeats statistical analysis 10,000 times with different data samples
- **Purpose**: Provides robust confidence intervals for research findings
- **Real-world use**: Validating survey results, ensuring statistical reliability

**Why it matters**:
- **Economics**: Risk assessment, portfolio optimization
- **Psychology**: Statistical validation of research findings
- **IT Support**: Demonstrates complex computational workflows
- **Cluster Admins**: Shows GPU efficiency for iterative calculations

**Expected results**: 75,000 Monte Carlo simulations completed in seconds

**Learn more**:
- [Monte Carlo Methods Explained](https://www.investopedia.com/terms/m/montecarlosimulation.asp)
- [Financial Risk Modeling](https://www.nvidia.com/en-us/data-center/gpu-accelerated-applications/financial-services/)
- [Bootstrap Methods in Statistics](https://en.wikipedia.org/wiki/Bootstrapping_(statistics))
- [GPU-Accelerated Monte Carlo](https://developer.nvidia.com/blog/accelerating-monte-carlo-simulations-with-gpus/)

### **Phase 4.5: Social Science Simulations**

#### **Agent-Based Models: Social Contagion**
**What it does**: Simulates how information or behaviors spread through social networks

**Real-world analogy**:
- Imagine modeling how a rumor spreads through a high school
- Each student (agent) can influence their friends
- We simulate 100,000 students over 100 time steps
- Shows how quickly information reaches the entire population

**Real-world applications**:
- **Sociology**: Understanding social movements and cultural diffusion
- **Psychology**: Modeling behavior change and social influence
- **Public Health**: Predicting disease spread and vaccination campaigns
- **Marketing**: Understanding viral marketing and product adoption

**Expected results**: 100,000 agents simulated in seconds

**Learn more**:
- [Agent-Based Modeling](https://en.wikipedia.org/wiki/Agent-based_model)
- [Social Network Analysis](https://en.wikipedia.org/wiki/Social_network_analysis)
- [GPU-Accelerated Social Simulations](https://developer.nvidia.com/blog/accelerating-social-science-research-with-gpus/)
- [NetLogo Agent-Based Modeling](https://ccl.northwestern.edu/netlogo/)

#### **Network Analysis: Community Detection**
**What it does**: Analyzes large social networks to identify communities and connections

**Real-world analogy**:
- Imagine analyzing Facebook friendships to find groups of people who know each other
- We analyze networks with 20,000 people and their connections
- Calculates how connected each person is and how tightly knit communities are

**Real-world applications**:
- **Sociology**: Understanding social structures and community formation
- **Anthropology**: Analyzing kinship networks and cultural groups
- **Political Science**: Mapping political alliances and influence networks
- **Business**: Identifying key influencers and opinion leaders

**Expected results**: 20,000 node networks analyzed in seconds

**Learn more**:
- [Network Analysis Fundamentals](https://en.wikipedia.org/wiki/Network_theory)
- [Social Network Analysis Tools](https://en.wikipedia.org/wiki/Social_network_analysis_software)
- [Community Detection Algorithms](https://en.wikipedia.org/wiki/Community_structure)
- [GPU-Accelerated Graph Analytics](https://developer.nvidia.com/blog/accelerating-graph-analytics-with-gpus/)

#### **Survey Research: Factor Analysis**
**What it does**: Analyzes large-scale survey data to identify underlying patterns

**Real-world analogy**:
- Imagine you have a 50-question survey completed by 100,000 people
- Factor analysis finds which questions measure the same underlying concept
- For example, questions about "happiness," "life satisfaction," and "optimism" might all measure "well-being"

**Real-world applications**:
- **Psychology**: Developing and validating psychological scales
- **Sociology**: Understanding social attitudes and beliefs
- **Political Science**: Analyzing voter preferences and political attitudes
- **Education**: Assessing student learning and engagement

**Expected results**: 100,000 respondent surveys analyzed in seconds

**Learn more**:
- [Factor Analysis in Psychology](https://en.wikipedia.org/wiki/Factor_analysis)
- [Survey Research Methods](https://en.wikipedia.org/wiki/Survey_methodology)
- [Psychological Scale Development](https://en.wikipedia.org/wiki/Psychometrics)
- [GPU-Accelerated Statistical Analysis](https://developer.nvidia.com/blog/accelerating-statistical-computing-with-gpus/)

#### **Political Science: Voting Behavior Simulation**
**What it does**: Simulates elections with realistic voter preferences and candidate positions

**Real-world analogy**:
- Imagine simulating an election with 200,000 voters and 5 candidates
- Each voter has preferences on 10 different issues (economy, healthcare, etc.)
- Each candidate has positions on these same issues
- Voters choose candidates based on how well their preferences match

**Real-world applications**:
- **Political Science**: Understanding electoral dynamics and voter behavior
- **Sociology**: Analyzing political polarization and social cleavages
- **Economics**: Modeling policy preferences and voting incentives
- **Public Policy**: Predicting election outcomes and policy impacts

**Expected results**: 200,000 voter elections simulated in seconds

**Learn more**:
- [Voting Systems and Electoral Analysis](https://en.wikipedia.org/wiki/Voting_system)
- [Political Science Research Methods](https://en.wikipedia.org/wiki/Political_science)
- [Spatial Voting Models](https://en.wikipedia.org/wiki/Spatial_voting_theory)
- [Computational Social Science](https://en.wikipedia.org/wiki/Computational_social_science)

## 🔧 **Technical Details for IT Support**

### **System Requirements**
- **GPU**: NVIDIA H100 or compatible GPU
- **Memory**: 128GB system RAM recommended
- **Storage**: 10GB free space for temporary files
- **Network**: High-speed interconnect for multi-node tests

### **Performance Expectations**
- **Matrix Operations**: 10-50 TFLOPS (trillion operations per second)
- **Memory Bandwidth**: 3-4 TB/s (terabytes per second)
- **Speedup vs CPU**: 50-100x on parallel workloads
- **Memory Usage**: Up to 80GB GPU memory for largest simulations

### **Troubleshooting Common Issues**

#### **"CUDA not available"**
- **Cause**: GPU drivers not installed or CUDA not configured
- **Solution**: Install NVIDIA drivers and CUDA toolkit
- **Check**: Run `nvidia-smi` to verify GPU detection

#### **"Out of memory"**
- **Cause**: GPU memory insufficient for large simulations
- **Solution**: Reduce simulation sizes or use memory-efficient settings
- **Check**: Monitor GPU memory with `nvidia-smi`

#### **"Partition not available"**
- **Cause**: H100 partition not configured or no access
- **Solution**: Verify partition configuration and user permissions
- **Check**: Run `sinfo -p H100` to see partition status

### **Monitoring and Optimization**
- **GPU Utilization**: Monitor with `nvidia-smi` or `gpustat`
- **Memory Usage**: Track with `nvidia-smi -l 1`
- **Temperature**: Ensure proper cooling for sustained performance
- **Power**: Monitor power consumption for efficiency

## 📈 **Benefits for Different Stakeholders**

### **For IT Support Staff**
- **Hardware Validation**: Verify GPU performance meets specifications
- **System Optimization**: Identify bottlenecks and configuration issues
- **Capacity Planning**: Understand resource requirements for different workloads
- **Troubleshooting**: Diagnose performance problems and memory issues

### **For Cluster Administrators**
- **Resource Allocation**: Optimize GPU scheduling and job placement
- **Performance Tuning**: Configure SLURM parameters for GPU workloads
- **Monitoring**: Set up alerts for GPU utilization and memory usage
- **Documentation**: Provide performance baselines for users

### **For Social Science Researchers**
- **Research Acceleration**: Reduce computation time from hours to minutes
- **Larger Studies**: Process bigger datasets and more complex models
- **Iterative Analysis**: Test multiple hypotheses quickly
- **Collaboration**: Share reproducible computational workflows

## 🎯 **Usage Examples**

### **Quick Performance Check**
```bash
# Run basic GPU validation
./h100_demo.sh
```

### **Detailed Benchmarking**
```bash
# Run all benchmarks with comprehensive reporting
./h100_demo.sh
# Review the detailed summary at the end
```

### **Troubleshooting**
```bash
# Check GPU status
nvidia-smi

# Monitor during demo
watch -n 1 nvidia-smi

# Check SLURM partition
sinfo -p H100
```

## 📝 **Key Takeaways**

### **Performance Highlights**
- **Massive Speedups**: 50-100x faster than CPU for parallel workloads
- **Large-Scale Processing**: Handle datasets with millions of records
- **Real-time Analysis**: Process complex models in seconds
- **Memory Efficiency**: Optimized for large-scale computations

### **Research Applications**
- **Economics**: Risk modeling, portfolio optimization, econometric analysis
- **Psychology**: Survey analysis, factor analysis, behavioral modeling
- **Sociology**: Network analysis, social dynamics, community detection
- **Political Science**: Electoral modeling, policy analysis, voting behavior

### **Infrastructure Benefits**
- **Scalability**: Handle growing computational demands
- **Efficiency**: Reduce energy consumption and compute time
- **Reliability**: Robust performance for production workloads
- **ROI**: Significant time savings for research teams

## 🤝 **Support and Documentation**

### **Getting Help**
- **Performance Issues**: Check GPU utilization and memory usage
- **Configuration Problems**: Verify SLURM settings and partition access
- **Research Applications**: Consult with domain experts for specific use cases

### **Further Resources**

#### **GPU Computing Fundamentals**
- [NVIDIA CUDA Documentation](https://docs.nvidia.com/cuda/) - Complete CUDA programming guide
- [PyTorch GPU Tutorials](https://pytorch.org/tutorials/beginner/blitz/tensor_tutorial.html) - GPU-accelerated deep learning
- [GPU Computing for Beginners](https://developer.nvidia.com/blog/even-easier-introduction-cuda/) - Easy introduction to GPU programming
- [HPC GPU Best Practices](https://developer.nvidia.com/blog/ten-easy-ways-to-resolve-cuda-launch-errors/) - Performance optimization tips

#### **Social Science Research Applications**
- [Computational Social Science](https://en.wikipedia.org/wiki/Computational_social_science) - Overview of computational methods in social sciences
- [GPU-Accelerated Research Papers](https://scholar.google.com/scholar?q=GPU+accelerated+social+science+research) - Academic research examples
- [Social Science Data Analysis](https://www.r-project.org/) - R programming for social science research
- [Python for Social Scientists](https://www.python.org/doc/) - Python programming resources

#### **HPC and Cluster Management**
- [SLURM Documentation](https://slurm.schedmd.com/documentation.html) - Job scheduling and cluster management
- [HPC Best Practices](https://www.nvidia.com/en-us/data-center/gpu-accelerated-applications/high-performance-computing/) - High-performance computing guidelines
- [Cluster Monitoring Tools](https://developer.nvidia.com/blog/monitoring-gpu-utilization-with-dcgm/) - GPU monitoring and diagnostics
- [HPC Community Forums](https://www.open-mpi.org/community/) - OpenMPI and HPC discussion groups

#### **Educational Resources**
- [Khan Academy Linear Algebra](https://www.khanacademy.org/math/linear-algebra) - Matrix operations and linear algebra
- [Statistics and Probability](https://www.khanacademy.org/math/statistics-probability) - Statistical concepts and methods
- [MIT OpenCourseWare](https://ocw.mit.edu/courses/mathematics/) - Free mathematics and statistics courses
- [Coursera Data Science](https://www.coursera.org/browse/data-science) - Online data science courses

#### **Research Communities**
- [Association for Computing Machinery](https://www.acm.org/) - Professional organization for computing
- [American Statistical Association](https://www.amstat.org/) - Statistics and data science community
- [Society for Political Methodology](https://polmeth.org/) - Political science methodology
- [International Association for Social Science Information Services](https://www.iassistdata.org/) - Social science data services

#### **Software and Tools**
- [R Statistical Computing](https://www.r-project.org/) - Statistical computing environment
- [Python Scientific Stack](https://scipy.org/) - Scientific computing with Python
- [Jupyter Notebooks](https://jupyter.org/) - Interactive computing environment
- [GitHub GPU Projects](https://github.com/topics/gpu-computing) - Open-source GPU computing projects

This comprehensive demo showcases how H100 GPUs can transform computational research across the social sciences while providing valuable insights for IT infrastructure management. The resources above provide pathways for deeper learning and practical application of GPU computing in research and system administration.
