from ppcgrader.info_utils import *

code = "llm"
name = "LLM"
descr = "large language model"


def html():
    from markupsafe import Markup
    return Markup(f"""
<p>In this exercise, you will optimize an implementation of a <em>Large Language Model</em>, in this particular case models
of the <a href="https://en.wikipedia.org/wiki/LLaMA">LLaMA</a> family. A short description of the connection between
parallel computation and the success of transformer-based language models such as LLaMA is given <a href="#hardware-lottery">below</a>.</p>

<p>Generally, the work of an LLM can be split into two phases: processing the user prompt and generating new tokens based on the prompt.
In this exercise, we concentrate on the former.
Already with only the user prompt processing, we can get interesting applications, like predicting how surprising a given piece of text is; see <a href="#demo">demo section below</a> for more information.</p>



<p>For this exercise, we have prepared a baseline implementation of the LLaMA 2 model, based on Andrej Karpathy's 
<a href="https://github.com/karpathy/llama2.c">llama2.c</a>. Your task is to identify the bottlenecks, 
and apply the techniques you have learned in this course to speed up prompt processing.</p>

<h3>Interface</h3>

<p>You need to implement the following function:</p>
<div class="prewrap"><pre>
void llm(LLamaConfig config, 
         LLamaParameters params, 
         const std::vector<token_t>& tokens, 
         std::vector<float>& logits);
</pre></div>
<p>The parameters are as follows:</p>
<ul>
<li><code>config</code> A struct that describes the configuration of the LLaMA model, in particular the shapes of the weight
matrices. See <code>llm.h</code> for the definition of this struct.</li>
<li><code>params</code> This struct contains the actual weights of the model.</li>
<li><code>tokens</code> The sequence of tokens that serve as the prompt for the model.</li>
<li><code>logits</code> This array needs to be filled in with the predicted scores (logits).</li>
</ul>

<p>You can assume that the configuration always describes a valid network (i.e., all sizes > 0). You may assume that 
network dimensions (<code>config.dim</code> and <code>config.hidden_dim</code>) will be a multiple of 16.</p>

<h3>Details</h3>
<p>The exercise comes with a baseline implementation that handles single-threaded, scalar processing of the LLaMA model.
For each token, this consists of the following steps
<ul>
<li>Take the numerical token id and select the corresponding embedding vector as the initial 
<code>dim</code>-dimensional hidden state of the network.</li>
<li>For each transformer block, update the hidden state of the network:</li>
<ul>
<li>Rescale the inputs using <a href="https://proceedings.neurips.cc/paper/2019/hash/1e8a19426224ca89e83cef47f1e7f53b-Abstract.html">RMSnorm</a>
with scale parameters <code>rms_attention</code>.</li>
<li>Calculate <code>query</code>, <code>key</code>, and <code>value</code> vectors for the attention mechanism using a matrix multiplication
with <code>query_weight_matrix</code>, <code>key_weight_matrix</code>, and <code>value_weight_matrix</code> respectively.</li>
<li>Adjust <code>query</code> and <code>key</code> using <a href="https://www.sciencedirect.com/science/article/abs/pii/S0925231223011864">rotary positional embeddings</a>.
<li>For each attention head, calculate the similarity between <code>key</code> and <code>value</code> at each past token 
using an inner product. Convert these into scores that sum to one using the 
<a href="https://en.wikipedia.org/wiki/Softmax_function">softmax</a> function.
<li>Calculate the weighted sum of all <code>value</code> vectors using the scores as weights.</li>
<li>Concatenate the results of each attention head and perform one final matrix multiplication with
 <code>out_weight_matrix</code>, add the result to the hidden state.</li>
<li>Generate a new, rescaled version of the hidden state using <code>rmsnorm</code>.</li>
<li>Project this state into two separate larger vector spaces of dimension <code>hidden_dim</code> by multiplying with the
weight matrices <code>feed_forward_w1</code> and <code>feed_forward_w3</code>. Combine the two vectors using <a href="https://arxiv.org/abs/2002.05202">SwiGLU</a>
pointwise operation.</li>
<li>Project back to original size <code>dim</code> using weight matrix <code>feed_forward_w2</code>, add the result to the hidden state.</li>
</ul>
<li>Perform one final normalization using <code>rmsnorm</code> with scaling parameters <code>RmsFinal</code>.</li>
<li>Calculate scores for each possible next token using the <code>num_tokens × dim</code> output weight matrix <code>TokenOutputMatrix</code></li>
<li>Convert to probabilities using <code>softmax</code>.</li>
</ul>
</p>

<p>
We provide implementations of these utilities in the <code>llm.h</code> header. You are free to use these functions,
or provide your own implementation if you think this will improve performance.
</p>

<h3 id="hint">Hints</h3>

<div class="spoiler">
    <p>Our tests contain setups in which specific submodules of the Transformer are disabled. These might help tracking down
    implementation errors</p>
    <p>Familiarize yourself with the provided reference implementation. Try to localize where most of the time is spent,
    and which parts of the computation are easily parallelizable. Just adding <code>#pragma omp parallel for</code> to the
    right places will give you at least one point for this exercise.</p>
    <p>Further improvements are possible if you apply ILP and vectorization, but to achieve full points in this exercise, 
    you need to come up with a more efficient parallelization strategy.</p>
</div>

<h3 id="demo">Demo</h3>
<p>
While generally, their ability to generate human-like text is what makes LLMs impressive, there are also interesting
things that can be done purely by prompt processing. In particular, since we can compare the predicted probability
distributions with the actual tokens in the text, we can find out which parts of the prompt were most surprising to
the network.
</p>

<p>In the exercise template, we provide a demo tool, invoked with <code>./grading demo</code>, that visualizes how surprising each token in the given text is.
For example, running <code>./grading demo "Once upon a time, there was a test story."</code> produces the following output:</p>

<div class="prewrap">
    <pre><span>Once</span><span style='color: darkgreen'> upon a time, there was a</span><span style='color: red'> test story</span><span style='color: darkgreen'>.</span></pre>
</div>

<p>Note that this involves downloading <a href="https://huggingface.co/karpathy/tinyllamas">a pre-trained network from online</a>.</p>

<h3 id="hardware-lottery">Autoregressive Language Modelling, Transformers, and Parallelism</h3>
<p>
Modern machine learning can achieve amazing results when given a large amount of <em>labeled</em> training data.
However, most existing data is unlabeled, and still contains valuable information that can be learned within its structure.
This is what an autoregressive language model is trying to capture: Given a sequence of words, predict which word is
going to be the next one. More generally, the text is split into tokens, which could be words, syllables, or even single letters.
All that is needed for training a task like this is a large corpus of text, which can be gathered by,
e.g., scraping Wikipedia. 
</p>

<p>
Possibly the simplest way to model this task is to apply a Markov assumption: The next element in the sequence only 
depends on the current element. In that case, the language model is just a huge probability table — given this current
token, these are the probabilities for the next token. While this is extremely efficient, it is also quite limited
in its modeling capacity. It can be used, for example, to generate fake words that capture the characteristics of an
actual language, but it cannot generate coherent sentences.
</p>

<p>
To increase expressibility, one could lengthen the context window of the Markov model. Instead of considering just the
last token, consider the last <code>n</code> tokens. Unfortunately, this makes the memory consumption (and required training data)
grow exponentially with the context size. An alternative is to compress the context into a <code>state</code> vector,
and make predictions based on that state. Conceptually
<div class="prewrap"><pre>
for token in sequence do
    state = update_state(state, token)
    prediction = output_prediction(state)
</pre></div>
  
<p>
To further improve expressibility, in typical deep-learning fashion, one can then stack multiple of these processes 
together in layers, such that the <code>state[l, p]</code> of layer <code>l</code> at location <code>p</code> in the sequence,
is a function of the state of the previous time-step, and of the previous layer at the same time step, 
<code>state[l, p] = update_state(state[l, p-1], state[l-1, p])</code>. 
</p>

<p>
There are several challenges when trying to use this approach for longer sequences; in particular, the function <code>update_state</code>
needs to be carefully chosen to allow signals to propagate over long time distances. If you are interested in more, 
<a href="https://karpathy.github.io/2015/05/21/rnn-effectiveness/">Andrej Karpathy</a> has a nice summary on the success
of recurrent neural networks.
</p>

<p>
Unfortunately, one challenge remains: How to train these networks with the vast amounts of data available? The structure
of the network update means that there exist dependencies both towards the preceding layer, and towards the previous state
within the same layer. This is in contrast to the more recent <a  href="https://en.wikipedia.org/wiki/Transformer_(machine_learning_model)">Transformer</a>
architecture, in which the next state is calculated based on <em>all</em> past states of only the previous layer, removing
one axis of dependencies. Consequently, while processing different layers has to be done sequentially, all the positions
within a layer can be handled in parallel. As a result, we now have large language models trained on 
<a href="https://github.com/togethercomputer/RedPajama-Data">trillions</a> of tokens.
</p>

<p>
The phenomenon that the success and failure of a machine learning method can be strongly dependent on the capabilities
of the current hardware has become known as the <a href="https://hardwarelottery.github.io/">Hardware Lottery</a>.
</p>
""")


def explain_web(raw: dict):
    templ_basic = """
{% if input.config %}
    <p>In this test I called your function with the following model:</p>
    <ul class="compact">
    {% for param, value in input.config.items() %}
         <li>ny = {{ param }} = {{value}}</li>
    {% endfor %}
    </ul>
    
    {% if input.length %}
         It processed a sequence of {{input.length}} tokens.
    {% endif %}
{% endif %}
{% if safenum(oe.max_error) > 0 %}
    <p>The predicted probabilities do not match with the reference probabilities.
    The largest mismatch happened at position {{ oe.location_pos }} for token {{ oe.location_tok }}.
    </p>
    <p>In comparison with the expected output, the largest error was ≈ <strong>{{ safereadable(oe.max_error) }}</strong>.
    In this test the largest errors should be &lt; {{ safereadable(oe.threshold) }}.
    That is, your code made errors that are ≈ {{ safereadable(oe.max_error/oe.threshold) }} times too large.</p>
    {% if saferatio(oe.max_error, oe.threshold, 10) %}
        <p>As the errors were relatively small, could they be rounding errors?</p>
    {% endif %}
{% endif %}
"""
    return render_explain_web(templ_basic, raw)


def explain_terminal(r, color=False):
    input = r.input_data or {}
    output = r.output_data or {}
    oe = r.output_errors or {}

    config = input.get('config', {})
    length = input.get('length', None)

    max_error = oe.get("max_error", None)
    location_pos = oe.get("location_pos", None)
    location_tok = oe.get("location_tok", None)
    threshold = oe.get("threshold", None)

    if color:
        hl, minor, reset = '\033[31;1m', '\033[34;1m', '\033[0m'
    else:
        hl, minor, reset = '', '', ''

    expl = ''
    if len(config) != 0:
        expl += f'In this test I called your function with the following model:\n'
        for param, value in config.items():
            expl += f' · {param} = {value}\n'
        expl += '\n'

        if length != 0:
            expl += f'It processed a sequence of {length} tokens.'

    if max_error:
        expl += '\n\n'
        expl += f"The predicted logits do not match the reference logits.\n"
        expl += f"The largest mismatch happened at position {location_pos} for token {location_tok}.\n"
        if max_error is not None and threshold is not None:
            expl += f'In comparison with the expected output, the largest error was ≈ {hl}{readable(max_error)}{reset}.\n'
            expl += f'In this test the largest errors should be < {readable(threshold)}.\n'
            rel = max_error / threshold
            expl += f'That is, your code made errors that are ≈ {readable(rel)} times too large.\n'
            if rel < 100:
                expl += f'As the errors were relatively small, could they be maybe rounding errors?\n'
        expl += '\n'
    return expl
