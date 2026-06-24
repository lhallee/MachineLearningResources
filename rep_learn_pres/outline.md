I am presenting a 45 minute talk to a high school math summer camp. My research pertains to representation learning for protein annotation and design, so I thought I would do a short talk on machine learning, intuition behind it, and basic representation learning, leading into some exciting topics like transformers and ChatGPT.

Here's the outline I was thinking of

- Sometimes you want to train some function f(x) to predict some valuable y
- There are many many many use cases for such functions
- How this is typically done is machine learning
- Sounds confusing or complicated, but the "machine" is a mathematical function with parameters and the "learning" is tuning them to predict y_hat the closest to y as possible
- In other words, we are trying to minimize the error in predictions
- Turns out, we can format this problem as a neural network
- Neural networks are just matmuls, which are just stacked vector dot products
- Here's some intuition about vector similarity heuristics, including dot products
- You can use the gradient to minimize a loss
- Here's a 2 parameter 2d visualization
- Most problems require more than 2 parameters, meaning more input features
- Some problems have some pretty straightforward features, like predictin the price of a car
- This is a classical example
- There are lots of easy numerical features to choose, like horsepower
- Some features are clearly not useful, like the number of wheels. Most cars almost always have 4 wheels, so the variance is essentially 0. Wheel size might be helpful though!
- You can use these features to predict the price with neural networks
- Some features aren't inherently numbers, like color of the car
- However, you can encode these as numbers, via integer assignment or one hot encoding
- However, these are not "semantic"
- Semanticity is important for learning dynamics and performance, as well as interpretability
- What if we could learn semantic features as a part of the neural network? Bake it into the gradient descent?
- Here is where repesentation learning comes in!
- One type of representation learning is embedding
- Here, we can embed each color, and every other categorical variable into a 16 length vector, randomly initialized as trainable weights.
- Now, we can minimize error with this in mind.
- Look, the model performs better now!
- Look, we can plot projections of the embeddings, similar colors are closer! Neat!
- This is the first step of transformer! Words are categorical variables well suited for embedding, we have no idea which numbers should represent them.
- Transformer can be trained to do various language tasks.
- Afterwards, look, king - man is similar to the royalty vector, royalty + woman is similar to the queen vector. Show more examples of this stuff too. Semantics!
- I apply these to protein language models
- Here's a demo (I will provide this)
- I hope you can see how interesting and useful representation learning is

I'm looking for you to
- Find a car price prediction dataset
- Try rigorous pytorch neural network training schemas with real validation and test splits to show that categorical variable embeddings can be more performant than integer encoding or one hot encoding.
- Find a small language model like ModernBERT-base to do king - man stuff.
Do all the experiments, come up with excellent simple visuals for everything, and produce the entire slide show. Use a common slide show template that is easy for you to edit and polish in powerpoint.
Use subagents extensively to parallelize the work.