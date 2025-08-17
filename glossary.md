# A

# **AGI (Artificial General Intelligence)**
 Categories: AI/ML Concepts, Safety & Alignment, Governance & Privacy

A hypothetical form of AI that can understand, learn, and adapt across essentially any intellectual task a human can tackle—not just narrow, pre-defined problems. It implies robust transfer learning, self-direction, and commonsense reasoning. Because no system today clearly meets that bar, debates around AGI focus as much on governance and safety as on technical feasibility.

# AI (Artificial Intelligence)

 Categories: AI/ML Concepts

Any system that performs tasks we associate with human intelligence—recognizing patterns, understanding language, making predictions—by learning from data rather than being explicitly hard‑coded for every rule. “AI” ranges from simple classifiers to massive multimodal models and agents that call external tools.

# AI Agent

 Categories: AI/ML Concepts, Agents & Tools

Software that perceives an environment (via inputs or tools), chooses actions to pursue a goal, and executes those actions—often in loops of plan → act → observe. Agents can be as simple as a chatbot that calls an API or as complex as a system that schedules tasks and manipulates files autonomously.

# AI Assistant

 Categories: Applications & Assistants

A conversational interface—text or voice—that helps users get things done: answer questions, draft content, operate apps, or automate tasks. Modern assistants can remember context, adapt tone, and invoke tools (search, email, spreadsheets) to deliver results instead of just text.

# AI Browser

 Categories: AI Devices, Applications & Assistants

A browser or browser layer that embeds AI-powered features like page summarization, inline question answering, and form filling. The label is marketing-driven; under the hood it’s usually a standard browser plus an LLM-backed sidebar or extension.

# AI Ethics

 Categories: Ethics & Explainability, Governance & Privacy

The study and practice of building and deploying AI systems fairly, transparently, and safely. It covers bias mitigation, privacy, explainability, accountability, environmental impact, and long-term societal effects.

# AI Phone

 Categories: AI Devices

A smartphone marketed around on-device or tightly integrated AI features: local transcription, summarization, photo and video editing, or small language models that run without the cloud. Most still fall back to cloud inference for heavy tasks.

# AI Safety

 Categories: Safety & Alignment, Governance & Privacy

Research and engineering work ensuring advanced AI systems remain aligned with human values, robust against misuse, and controllable—even as capabilities scale rapidly.

# AI Search

 Categories: Retrieval & Search, Applications & Assistants

Search engines that parse natural-language questions and synthesize answers (often with citations) instead of just listing links. They blend retrieval, LLM generation, and sometimes live web browsing.

# Algorithm

 Categories: AI/ML Concepts

A clear, step-by-step recipe a computer follows to solve a problem. In machine learning the algorithm specifies how the model learns from data.

# Alignment

 Categories: Safety & Alignment

Ensuring an AI system actually pursues human-intended goals and respects constraints as its capabilities grow. It spans technical work (e.g., RLHF, constitutional prompts) and governance policies to keep systems steerable and safe.

# **Anthropic**

 Categories: Companies & Organizations

An AI lab founded by former OpenAI researchers, known for the Claude model family and its “Constitutional AI” approach—using written principles to shape behavior during training.

# API

 Categories: Infrastructure & Hardware

A set of rules allowing one program to talk to another. AI APIs let developers add abilities such as image recognition or text generation without training their own model.

# ASR (Automatic Speech Recognition)

 Categories: Speech & Voice

Software that converts spoken audio into text by mapping acoustic signals to phonemes/words and decoding them with a language model. Modern ASR systems (often transformer-based) handle accents, noise, and code‑switching, power voice assistants and call analytics, and are sometimes called “speech‑to‑text.”

# Attention Mechanism

 Categories: Model Architecture

A neural component that lets models weight which input tokens matter most at each step—akin to selectively “paying attention.” It’s the core ingredient that made Transformers scale to long sequences efficiently.

# B

# Benchmark

 Categories: Evaluation & Benchmarks

A standardized dataset + metric used to compare models (e.g., MMLU for reasoning, ImageNet for vision). Claims of “state-of-the-art” almost always reference specific benchmarks, so context and dates matter.

# Bias

 Categories: Ethics & Explainability

Systematic, unfair skew in model outputs—often inherited from imbalanced or prejudiced training data. Detecting, measuring, and mitigating bias is central to ethical, trustworthy AI deployment.

# Big Data

 Categories: Data, Infrastructure & Hardware

Vast, fast, or varied datasets that overwhelm traditional processing. ML thrives on big data because scale surfaces subtle patterns—but also amplifies any embedded biases.

# Bounding Box

 Categories: Computer Vision

The rectangle a vision model draws around a detected object to localize it in an image—fundamental to object detection pipelines and downstream tasks like tracking.

# C

# Chain-of-thought (CoT)

 Categories: Reasoning, Prompting

Prompting or training a model to spell out its intermediate reasoning steps (e.g., “let’s think step by step”) before giving the final answer, which often boosts accuracy on multi‑step problems. These traces aren’t guaranteed to reflect the model’s true internal logic and can leak sensitive info, so many production systems use hidden CoT or distill the benefits into shorter, safer rationales.

# ChatGPT

 Categories: Language (LLMs & NLP), Applications & Assistants

OpenAI’s conversational interface built on the GPT family. It answers questions, writes and edits text or code, and can call tools like web browsers or data APIs. Newer versions handle images, audio, and very long contexts, effectively acting as a general-purpose assistant rather than a simple chatbot.

# Chats

 Categories: Applications & Assistants

In AI, "chats" refer to the interactions or conversations between a user and a conversational agent (such as a chatbot or virtual assistant). Chats can encompass a variety of formats, including text-based messaging, voice communication, or even visual inputs. The focus is on the exchange of information, where users can ask questions, seek assistance, or engage in dialogue.

# Claude

 Categories: Language (LLMs & NLP), Companies & Organizations

Anthropic’s large language model/assistant, designed around “helpful, harmless, honest” principles. Claude handles long documents, images, and complex reasoning, and can follow constitutional rules baked into training to keep outputs within desired bounds.

# Classification

 Categories: Training & Learning

A supervised learning task where a model learns from labeled examples to assign new inputs to one of several predefined categories (e.g., “spam” vs. “not spam”). In LLM-era workflows, you often prompt a general model to choose a label instead of training a separate classifier, trading a bit of control for speed and flexibility.

# Cloud Computing

 Categories: Infrastructure & Hardware

Renting compute, storage, and networking from remote data centers (AWS, GCP, Azure) instead of buying and maintaining your own hardware. Most large-model training and high-availability inference live in the cloud because it’s elastic, global, and integrates with managed ML ops tools.

# Clustering

 Categories: Training & Learning

An unsupervised technique that groups items so that members of the same cluster are more similar to each other than to those in other clusters—no labels required. It’s handy for organizing documents, discovering customer segments, or spotting anomalies after you’ve embedded everything into vectors.

# Computer Vision

 Categories: Computer Vision

The field (and set of models) that let machines interpret visual data—detecting objects, segmenting scenes, captioning images, or tracking movement in video. Modern vision systems often use CNNs or Vision Transformers, and increasingly pair vision with language for multimodal tasks.

# Confidence Score

 Categories: Evaluation & Benchmarks

A numeric estimate of how sure a model is about its output. These scores can be miscalibrated (high confidence but wrong), so teams often apply calibration methods or secondary checks before acting on them in production.

# Context Window

 Categories: Language (LLMs & NLP)

The maximum number of tokens an LLM can “see” at once. Bigger windows let you stuff in long conversations or entire PDFs, but at the cost of more memory and slower inference—so retrieval and summarization are used to manage context efficiently.

# Conversational AI

 Categories: Applications & Assistants

Systems that hold multi-turn dialogues over text or voice, tracking context, clarifying intent, and often invoking external tools to actually complete tasks. It spans simple FAQ bots to full agent frameworks that schedule meetings or run code.

# CPU (Central Processing Unit)

 Categories: Infrastructure & Hardware

The general-purpose processor in a computer. CPUs juggle many different tasks well, but deep learning’s heavy matrix math typically runs faster on GPUs or specialized accelerators; CPUs still orchestrate, preprocess, and handle I/O around those workloads.

# D

# Data Center
 Categories: Infrastructure & Hardware

A facility filled with servers, networking gear, and cooling systems where training runs, vector databases, and AI APIs physically live. Hyperscalers design them for massive power draw and redundancy; model cost and latency are tightly tied to where (and how efficiently) your workloads run.

# Data Mining
 Categories: Data

The classic practice of discovering patterns and relationships in large datasets using statistics and machine learning. It predates today’s deep learning boom and still underpins tasks like churn prediction, market basket analysis, and anomaly detection.

# Data Science
 Categories: Data

The end‑to‑end craft of turning raw data into useful decisions—collecting, cleaning, exploring, modeling, evaluating, and communicating results. In AI teams, data scientists often prototype models, validate performance, and translate findings into business impact.

# Decision Trees
 Categories: Training & Learning

A supervised‑learning model that makes decisions by recursively splitting data along feature thresholds, creating a tree of “if‑then” rules that ends in leaf nodes with class labels or numeric predictions. It’s easy to interpret—trace the path from root to leaf—but prone to overfitting unless pruned or averaged in ensembles like Random Forests and Gradient‑Boosted Trees.

# Deep Learning
 Categories: AI/ML Concepts

A subset of machine learning that uses multi-layer neural networks to learn features directly from raw data (text, pixels, audio) instead of hand-engineered rules. Depth and scale let these models capture extremely complex patterns, but they demand huge compute and careful optimization.

# DeepSeek
 Categories: Companies & Organizations, Language (LLMs & NLP), Open Source

A Chinese research group releasing open-source LLMs spanning code, reasoning, and general chat. It’s grown beyond the original “DeepSeek-Coder,” shipping newer reasoning-focused variants and math-heavy models that compete with Western open weights.

# Diffusion Model
 Categories: Generative AI

A generative approach that starts from random noise and iteratively denoises it to produce an image, audio clip, or even video. Stable Diffusion, DALL·E 3, and many cutting-edge video systems use diffusion because it yields high fidelity and fine control.

# Distillation
 Categories: Model Optimization

Training a smaller “student” model to mimic a larger “teacher” model’s behavior—usually by matching its logits, soft labels, or chain‑of‑thought traces—so you keep most of the performance with far fewer parameters and cheaper inference. Variants include offline (pretrained teacher), online (teacher and student learn together), and self‑distillation (the model teaches itself); it’s now standard for compressing LLMs, speeding deployment, and packaging “reasoning” into lighter weights.

# E

# Embedding
 Categories: Embeddings & Vectors, Retrieval & Search

A dense numeric vector learned by a model to represent an object—text, image, audio, user profile—so that semantic similarity becomes geometric closeness. Embeddings power semantic search, recommendation, clustering, and retrieval-augmented generation.

# ELIZA
 Categories: History, Applications & Assistants

ELIZA is a 1964-1966 MIT program that used pattern-matching scripts to mimic a Rogerian psychotherapist, becoming one of the first chatbots and inspiring the term “ELIZA effect” for users’ tendency to ascribe human traits to simple programs.

# Evaluation/Evals
 Categories: Evaluation & Benchmarks

The systematic process of testing what a model can (and cannot) do, using benchmarks, adversarial prompts, human preference studies, or domain-specific scorecards. Good evals catch regressions, uncover bias and hallucinations, and inform when a model is “safe enough” to ship.

# Explainable AI
 Categories: Ethics & Explainability

Methods that make a model’s behavior legible to humans—feature attributions, saliency maps, counterfactuals, simplified surrogate models, or structured rationales. For LLMs, explanations are often post hoc text and must be treated carefully, since fluency doesn’t guarantee faithfulness.

# F

# Feature
 Categories: Data

A measurable property or signal the model uses to make predictions—anything from a pixel value or timestamp to a dimension in an embedding. In deep learning, many “features” are learned automatically inside hidden layers rather than handcrafted. Good features concentrate task-relevant information and discard noise.

# Feature Extraction
 Categories: Data

Selecting or computing the most The process of turning raw data into informative features that a model can use efficiently. Classic ML relied on manual feature engineering (TF‑IDF for text, SIFT for images); modern pipelines often let pretrained models do the heavy lifting and then reuse their intermediate representations.

# Federated Learning
 Categories: Governance & Privacy, Training & Learning

A privacy‑preserving training approach where the model stays on user devices or edge servers; each client learns from its local data and sends only model updates (not raw data) to a central coordinator, which aggregates them into a global model. This allows collective learning across sensitive datasets—phones, hospitals, banks—while keeping personal data on‑device and reducing regulatory risk.

# Fine-tuning
 Categories: Training & Learning

Taking a large, pretrained model and giving it a short, targeted training pass on domain- or task-specific data so it adapts quickly. Fine-tuning can align tone, follow custom instructions, or specialize in legal/medical jargon—without the cost of training from scratch.

# Foundation Model
 Categories: Language (LLMs & NLP), Multimodal AI

A very large model trained on broad, diverse data (text, images, code, audio) so it learns general capabilities. Developers then adapt it via prompting, fine-tuning, or tool use for countless downstream tasks—hence “foundation.”

# G

# Gemini
 Categories: Language (LLMs & NLP), Multimodal AI, Companies & Organizations

Google’s multimodal model family that natively handles text, images, audio, and video in one system. It’s positioned as a generalist assistant and developer platform, with tiers optimized for mobile, enterprise, and high-end reasoning workloads.

# Generative AI
 Categories: Generative AI

Models that learn the statistical structure of data and then sample new content from that distribution—text, code, images, audio, video. They power everything from chatbots and design tools to data augmentation and simulation, often guided by prompts or control signals.

# GAN (Generative Adversarial Network)
 Categories: Generative AI, Computer Vision

A two-model setup where a generator creates candidates and a discriminator critiques them. Through this adversarial game, the generator learns to produce increasingly realistic samples (images, audio, etc.), though training can be unstable and mode collapse is common.

# GPU (Graphics Processing Unit)
 Categories: Infrastructure & Hardware

A highly parallel processor originally built for graphics but ideal for the matrix math in deep learning. GPUs dominate both training and high-throughput inference, while CPUs handle orchestration, preprocessing, and lighter workloads.

# GPT (Generative Pre-trained Transformer)
 Categories: Language (LLMs & NLP), Companies & Organizations

OpenAI’s series of large language models trained on internet-scale text and adapted via fine-tuning or prompting. “GPT” now shorthand for “powerful LLM,” though the family includes many sizes, capabilities, and multimodal variants.

# Gradient Descent
 Categories: Training & Learning

The core optimization procedure in deep learning—compute how wrong the model is (loss), find the gradient of that loss with respect to parameters, and nudge parameters downhill. Variants like Adam, RMSProp, and momentum speed convergence and stabilize training on huge models.

# Grok
 Categories: Language (LLMs & NLP), Applications & Assistants, Companies & Organizations

xAI’s chatbot built on the company’s in‑house LLMs, pitched as witty and “uncensored.” Beyond the personality, it showcases xAI’s push toward models that can browse live data on X (Twitter) and respond with a more irreverent tone than typical corporate assistants.

# H

# Hallucination

 Categories: Safety & Alignment, Language (LLMs & NLP)

When a model confidently generates content that looks plausible but is factually wrong or entirely made up—an artifact of predicting the “next likely token” rather than verifying truth. Techniques like retrieval grounding and stricter prompting help reduce (not eliminate) it.

# Hugging Face

 Categories: Companies & Organizations, Open Source

An open-source hub for models, datasets, and ML tools (notably the transformers library), plus hosting and inference endpoints. It’s the de facto GitHub of AI, where researchers and devs share weights, run evals, and build on each other’s work.

# Hunyuan

 Categories: Language (LLMs & NLP), Multimodal AI, Companies & Organizations

Tencent’s family of large models spanning text, vision, and video, optimized for long context and enterprise integration. Public technical details are sparse; think of it as Tencent’s internal “foundation” stack rather than a single architecture.

# Hybrid Architecture

 Categories: Model Architecture

A model design that mixes paradigms—e.g., Transformer layers with state-space or recurrent components—to capture long context efficiently, reduce compute, or add inductive biases a pure Transformer lacks.

# Hyperparameter

 Categories: Training & Learning

A training-time setting you choose before learning starts—learning rate, batch size, number of layers—that shapes how (and how well) the model trains. Tuning hyperparameters can mean the difference between convergence and a model that never learns.

# I

# Image Generation

 Categories: Generative AI, Computer Vision

Creating novel images from prompts, sketches, or reference pictures using diffusion or transformer-based models (DALL·E 3, Midjourney, Stable Diffusion). Modern systems offer style control, in/outpainting, and even video-temporal consistency extensions.

# Inference

 Categories: Deployment & MLOps

The act of running a trained model forward to obtain outputs for new inputs—no parameter weights are updated. For LLMs, inference covers everything from answering questions to generating long-form text, often under latency and cost constraints. Optimizing inference involves tricks like quantization, batching, caching key/value states, or using specialized accelerators.

# Input Token

 Categories: Tokens, Language (LLMs & NLP)

The smallest unit a model counts when consuming context—typically a subword chunk or a few characters, depending on the tokenizer. Billing and context limits are usually expressed in tokens, not characters. Because it’s part of the prompt/context, an input token is processed once and often priced lower than generated (output) tokens by API providers.

# K

# Knowledge Graph

 Categories: Retrieval & Search, Reasoning

A structured network of entities (people, places, concepts) and their relationships, enabling symbolic queries and factual reasoning. Paired with LLMs, it can ground generations in hard facts and let agents reason over explicit connections.

# L

# Latent Space

 Categories: Embeddings & Vectors

The internal, high-dimensional space in which a model represents concepts and relationships after training. Distances and directions in this space correspond to semantic relations (e.g., “king” – “man” + “woman” ≈ “queen” in word embeddings).

# Layer

 Categories: Model Architecture

A set of neurons operating at the same “depth” in a neural network. Each layer transforms its input representation before passing it on; stacking many layers lets models learn hierarchical features (edges → shapes → objects, characters → words → concepts).

# LLaMA

 Categories: Language (LLMs & NLP), Companies & Organizations, Open Source

Meta’s family of open‑weight large language models (LLaMA → Llama 2 → Llama 3/3.1…) released in multiple sizes (single‑digit to tens of billions of parameters) for research and commercial use. They’re instruction‑tuned, strong at code and reasoning for their size, and serve as the base for countless community fine‑tunes and derivative models—making Llama the de facto backbone of much of today’s open‑source LLM ecosystem.

# LLM (Large Language Model)

 Categories: Language (LLMs & NLP)

A neural network with billions (or trillions) of parameters trained on massive text corpora to predict the next token. With scale and instruction tuning, LLMs can follow prompts, reason over long context, write code, and act as general-purpose text (and often multimodal) interfaces.

# Loss Function

 Categories: Training & Learning

The math that scores how wrong a model’s prediction is on training data. Training is just minimizing this loss—tweaking parameters so the score drops—so the choice of loss (cross-entropy, MSE, contrastive) directly shapes what “success” means for the model.

# M

# Machine Learning
 Categories: AI/ML Concepts

Letting computers improve at a task by learning patterns from data rather than hard-coded rules. It spans supervised, unsupervised, and reinforcement paradigms, and underpins everything from spam filters to generative models and recommendation engines.

# Mamba Architecture
 Categories: Model Architecture

A newer sequence model based on state-space ideas that can handle very long contexts more efficiently than vanilla Transformers. It trades the quadratic attention cost for linear-time processing, making it attractive for long docs, logs, or DNA sequences.

# MCP (Model Context Protocol)
 Categories: Protocols & Standards, Agents & Tools

An open specification (kicked off by Anthropic and others) that standardizes how models discover and call external tools, files, or data sources. Instead of ad hoc glue code per vendor, MCP defines a clean request/response interface for “tool use” at scale.

# Metadata
 Categories: Data, Retrieval & Search

Data about your data—timestamps, authors, file types, tags—that helps with governance, filtering, and smarter retrieval. In RAG systems, rich metadata lets you route queries, enforce permissions, and rank results beyond pure text similarity.

# Mistral
 Categories: Companies & Organizations, Language (LLMs & NLP), Open Source

A French AI company shipping lean, high-performance open models (and a commercial API) that compete with larger labs on efficiency. Known for releasing strong small/medium LLMs and embracing an open-weights philosophy.

# Model
 Categories: Model Architecture

The learned function that maps inputs to outputs—its parameters encode everything it has absorbed from training data. “Model” can mean anything from a tiny logistic regression to a trillion-parameter multimodal transformer.

# Model Training
 Categories: Training & Learning, Deployment & MLOps

The compute-intensive process of feeding data through a model, computing loss, and updating parameters (via backpropagation/optimizers) until performance plateaus. It involves data curation, hyperparameter tuning, checkpointing, and lots of monitoring to avoid over/underfitting.

# Multimodal AI
 Categories: Multimodal AI, Applications & Assistants

Models that can understand and generate across more than one data type—text, images, audio, video—often in a single architecture. This lets them ground language in vision (describe an image, answer questions about a chart) or mix modalities in outputs (generate narrated videos). Training requires aligned datasets and careful balancing so one modality doesn’t dominate. The payoff is richer, more context-aware reasoning and interaction.

# N

# NLP (Natural Language Processing)
 Categories: Language (LLMs & NLP), Applications & Assistants

The field of AI devoted to enabling computers to understand, generate, and interact with human language. Traditional NLP focused on tasks like parsing, sentiment analysis, and machine translation with handcrafted features; modern NLP leans heavily on large pretrained transformers that can be adapted via prompting or fine-tuning. NLP underpins chatbots, search, summarization, and countless text analytics applications.

# Neural Network
 Categories: AI/ML Concepts

A stack of interconnected “neurons” (simple math units) whose connection weights are learned from data. Forward passes transform inputs through layers; backpropagation adjusts weights to reduce error. Depth and width let networks approximate extremely complex functions, at the cost of large datasets and compute. Despite the biological metaphor, today’s nets are engineered math, not mini-brains.

# Nvidia
 Categories: Companies & Organizations, Infrastructure & Hardware

The dominant supplier of GPUs and software (CUDA, cuDNN) that power modern deep learning. Its hardware (A100, H100, etc.) runs most large-scale training and high-throughput inference, and DGX/Grace systems target end-to-end AI stacks. Nvidia’s ecosystem lock-in (hardware + drivers + libraries) makes it a central player in AI economics and capacity planning.

# O

# OCR (Optical Character Recognition)
 Categories: Computer Vision, Applications & Assistants

Converting images or scans of text into machine-readable characters. Modern OCR pipelines first detect text regions, then recognize characters—often with CNNs or transformers—handling skew, noise, and multiple languages. OCR is essential for digitizing PDFs, invoices, forms, and historical documents so they can be searched or fed into downstream models.

# Open-source
 Categories: Open Source

Software (or model weights) released under a license that lets anyone inspect, use, modify, and redistribute it. In AI, open weights enable community audits, rapid iteration, and local/private deployment. Licenses vary—from permissive (Apache/MIT) to restrictive (non-commercial)—so “open” can mean different freedoms.

# OpenAI
 Categories: Companies & Organizations, Language (LLMs & NLP)

The lab behind GPT, ChatGPT, DALL·E, and Whisper, originally founded as a nonprofit with a mission to ensure AGI benefits all humanity. It now operates a capped-profit structure and offers commercial APIs while funding safety/alignment research. OpenAI helped popularize instruction tuning, RLHF, and tool-using assistants at scale.

# Output Token
 Categories: Tokens, Language (LLMs & NLP), Deployment & MLOps

A chunk of text the model generates during inference. Most APIs bill these separately (and often higher) than input tokens because generation can’t be batched as aggressively and involves more sampling logic. Output length impacts latency and cost, so prompt design and stop conditions matter operationally.

# Overfitting
 Categories: Training & Learning

When a model learns the idiosyncrasies and noise of the training data so thoroughly that it fails to generalize to new data. Symptoms include very high training accuracy but poor validation/test performance. Regularization, dropout, early stopping, and larger, more diverse datasets are common defenses.

# P

# Parameter
 Categories: Model Architecture

A learned weight (or bias) inside a model that determines how inputs are transformed as they pass through layers. Modern LLMs have billions or trillions of parameters, which store their learned “knowledge.” Parameter count affects capacity and cost, but architecture, data quality, and training strategy matter just as much.

# Perplexity
 Categories: Evaluation & Benchmarks, Language (LLMs & NLP)

In language modeling, a metric of how well a model predicts text—lower perplexity means the model finds the sequence less “surprising.” Separately, Perplexity is also the name of an AI search engine that answers questions with cited sources.

# Personalization
 Categories: Personalization, Applications & Assistants

Tailoring a model’s outputs to a specific user—drawing on their history, preferences, or context to adjust tone, recommendations, or content. It can happen at inference time (prompt/context conditioning) or via fine-tuning on user data, with privacy and bias considerations baked in.

# Prompt
 Categories: Prompting, Language (LLMs & NLP)

The instruction plus any context, examples, or constraints you feed an LLM to steer its behavior. Good prompts frame the task clearly, set the output format, and provide just enough signal to reduce ambiguity without overloading the context window.

# Prompt Engineering
 Categories: Prompting

The craft of systematically designing, testing, and iterating prompts (and tool-calling schemas) to get consistent, high-quality results from a model. It includes techniques like few-shot examples, chain-of-thought scaffolding, output validators, and guardrails that catch failures.

# Q

# Quantization
 Categories: Model Optimization, Deployment & MLOps

Compressing a model by storing weights/activations in fewer bits (e.g., 8‑, 4‑, even 2‑bit) to shrink memory and speed inference, with minimal accuracy loss if done well. Post-training quantization is fastest to apply; quantization-aware training can preserve more fidelity.

# R

# RAG (Retrieval-Augmented Generation)
 Categories: Retrieval & Search, Applications & Assistants

A pattern where the system fetches relevant documents (via embeddings or keyword search) and injects them into the prompt before the model answers. This grounds outputs in real data, reduces hallucinations, and allows updates without retraining the base model.

# Reasoning Model
 Categories: Reasoning, Language (LLMs & NLP)

Marketing shorthand for LLMs tuned (often via special datasets, chain-of-thought distillation, or reinforcement learning) to tackle multi-step logic problems. In practice, most top models use similar tricks; “reasoning” usually means better intermediate planning, not symbolic theorem proving.

# Reinforcement Learning
 Categories: Training & Learning

Training an agent to act in an environment by maximizing cumulative reward. Instead of labeled examples, it learns from trial and error—balancing exploration and exploitation—with algorithms like Q-learning or policy gradients. RL underpins robotics, game-playing AIs, and preference tuning (e.g., RLHF).

# RLHF (Reinforcement Learning from Human Feedback)
 Categories: Safety & Alignment, Training & Learning

Humans rank or score model outputs; a reward model learns those preferences, and the base model is then fine-tuned to produce answers people like—aligning behavior without hand-written rules.

# S

# Semantic Search
 Categories: Retrieval & Search

Search that uses meaning, not just exact keywords. You embed queries and documents into vectors and retrieve by semantic closeness, so “cheap flights to Tokyo in November” can match “budget airfare deals for Tokyo this fall.” It powers modern RAG stacks, FAQ bots, and recommendation engines where wording varies but intent is similar.

# Sentiment Analysis
 Categories: Language (LLMs & NLP), Applications & Assistants

Automatically classifying the emotional tone of text—positive, negative, neutral, or finer-grained moods. Classic pipelines used lexicons and classifiers; today, LLMs or fine-tuned transformers score sentiment across reviews, tweets, support tickets, and more. Calibration matters: sarcasm, domain slang, and mixed feelings trip up naive models.

# Sessions
 Categories: Applications & Assistants, Data

A "session" in AI refers to a defined period of interaction between a user and an AI system. It represents the duration from when a user initiates an interaction (e.g., starting a chat) until that interaction concludes (e.g., closing the chat or timing out). Sessions can include multiple chats and are often used to track user activity, preferences, and context over time. Key characteristics include:

- State Management: Sessions often involve maintaining user data and context throughout the interaction.
- Duration: Sessions can be short (one-time interactions) or long (extended dialogues).
- Session IDs: Unique identifiers are often generated to track individual sessions for analytics or personalization.

# Speech-to-Text
 Categories: Speech & Voice, Applications & Assistants

Converting spoken audio into written words (automatic speech recognition). Modern systems chunk audio, detect phonemes or tokens, and decode with neural language models, handling accents, noise, and code-switching far better than in the 2010s. High-quality ASR underpins voice assistants, call-center analytics, meeting transcription, and accessibility tools.

# SOTA (State-of-the-Art)
 Categories: Evaluation & Benchmarks

The best reported performance on a benchmark at a given time. Because benchmarks evolve and tricks overfit them, “SOTA” is a moving target—always ask “on what task, when, and under what constraints?” Treat SOTA claims as a snapshot, not a universal crown.

# Structured Reasoning
 Categories: Reasoning

Forcing or guiding a model to think in explicit steps—plans, chains of thought, decision trees—rather than one-shot answers. It can be prompt-based (“let’s reason step by step”), tool-enforced (scratchpads, program interpreters), or architectural (tree-of-thought, graph-of-thought). The goal is more reliable logic and easier debugging of failures.

# Supervised Learning
 Categories: Training & Learning

Training models on labeled examples where the correct answer is known. The model learns a mapping from inputs to outputs (class, number, sequence) and is evaluated on how well it generalizes to unseen data. Most practical ML—spam detection, fraud scoring, defect classification—still lives here.

# T

# Tensor
 Categories: AI/ML Concepts

The fundamental data structure in deep learning—an N-dimensional array (scalars, vectors, matrices, then “tensors” beyond). Frameworks like PyTorch and TensorFlow optimize tensor operations (sums, multiplies, convolutions) on GPUs/TPUs, so almost every forward/backward pass is just tensor math.

# Text-to-Image
 Categories: Generative AI, Computer Vision

Generating images purely from textual prompts using diffusion or transformer models (e.g., “a watercolor painting of a fox reading a book at dawn”). Modern systems let you control style, composition, aspect ratio, and can edit existing images (inpainting/outpainting) for design workflows and creative prototyping.

# Text-to-Speech
 Categories: Speech & Voice, Generative AI

Converting written text into natural-sounding audio. Modern TTS models capture prosody, pacing, and emotion, can clone specific voices, and even switch languages mid‑sentence. Latency, clarity in noisy environments, and disclosure of synthesized voices are key production concerns.

# Token
 Categories: Tokens, Language (LLMs & NLP)

The smallest unit an LLM processes—usually a subword chunk (e.g., “trans-”, “former”). Context limits and pricing are counted in tokens, not characters, so prompt length optimization often means trimming tokens. Different tokenizers split text differently, which can affect both cost and model behavior.

# Tool Use
 Categories: Agents & Tools

When an AI system calls external APIs, databases, or code runners to get facts or perform actions it can’t do internally. A “tool-aware” prompt or protocol (e.g., MCP, OpenAI’s tool calling) tells the model what’s available; the model decides when to call, then integrates results into its answer.

# Training Data
 Categories: Data, Training & Learning

The corpus a model learns from—web text, code, images, audio, proprietary docs. Data quality, diversity, and cleanliness heavily shape model capability and bias; licensing and privacy constraints govern what can be included.

# Transfer Learning
 Categories: Training & Learning

Reusing knowledge from one task/domain to accelerate another. Instead of training from scratch, you fine-tune or prompt a large pretrained model so it adapts quickly—crucial for niche domains (legal, medical) where labeled data is scarce.

# Transformer
 Categories: Model Architecture

A transformer is a neural-network architecture built entirely on self-attention (no recurrence) that processes sequences in parallel, first described in the 2017 paper “Attention Is All You Need,” and now underpins most modern language and multimodal AI models.

# Turing Test
 Categories: History, Evaluation & Benchmarks

A machine passes the Turing Test when, in a blind text conversation, a human judge cannot reliably tell its replies from those of another person, demonstrating human-level conversational behaviour.

# U

# Underfitting
 Categories: Training & Learning

A too-simple model that fails to capture patterns, performing poorly on both training and unseen data.

# Unsupervised Learning
 Categories: Training & Learning

Learning patterns, structures, or representations from unlabeled data. Classic examples include clustering and dimensionality reduction; more recent paradigms like self-supervised learning blur the line by creating proxy labels from the data itself. Unsupervised pretraining often provides strong foundations for downstream supervised tasks.

# V

# Vector
 Categories: Embeddings & Vectors

An ordered list of numbers that encodes a data point so it can be manipulated algebraically—added, averaged, compared with dot products or cosine similarity. In AI, vectors are the lingua franca for representing everything from words to images (via embeddings), making efficient search, retrieval, and mathematical reasoning possible. Unlike raw text or pixels, vectors enable fast, approximate nearest-neighbor lookups at scale.

# Vector Database
 Categories: Retrieval & Search, Infrastructure & Hardware

A database built to store embedding vectors and retrieve nearest neighbors fast (via ANN indexes like HNSW or IVF). It’s the backbone of semantic search and RAG pipelines, where you need millisecond lookups over millions of vectors.

# Video Generation (AI Video)
 Categories: Generative AI, Computer Vision

Generating coherent video clips from text/image prompts using diffusion or transformer pipelines. Modern systems handle motion consistency, camera moves, and scene transitions end‑to‑end—far beyond early “stitched frame” hacks.

# Vision Model
 Categories: Computer Vision, Model Architecture

Any model specialized in understanding visual data—classification, detection, segmentation, captioning. Increasingly, these are paired with language models for multimodal reasoning (ask-about-this-image).

# Voice AI
 Categories: Speech & Voice, Applications & Assistants

The umbrella for speech recognition, voice synthesis/cloning, and real-time voice assistants. It stitches ASR, LLMs, and TTS to let users talk to software naturally (and hear it talk back).

# W

# Weight
 Categories: Model Architecture

A learned parameter that controls how signals flow between neurons. Collectively, billions of weights encode everything the model “knows.”

# Whisper
 Categories: Speech & Voice, Open Source, Companies & Organizations

OpenAI’s open-source automatic speech recognition (ASR) model trained on ~680k hours of multilingual, noisy, and accented audio, so it transcribes and translates (to English) with strong robustness out of the box.

# X

# xAI
 Categories: Companies & Organizations

Elon Musk’s AI company aiming for “truth-seeking” models; its flagship assistant is Grok, which leans into a snarkier, live-web persona.

# Z

# Zero-shot Learning
 Categories: Training & Learning, Prompting

When a model solves a task or recognizes a class it never saw in training by leveraging general representations or instructions. It’s fast and data-free to deploy but usually lags behind few‑shot or fine‑tuned models and is sensitive to how you phrase the instructions.
