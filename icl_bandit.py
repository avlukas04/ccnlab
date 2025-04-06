# ICL Pipeline for Two-Armed Bandit Cognitive Modeling with Reversal Tasks

import random
import numpy as np
from openai import OpenAI
from typing import List, Dict
import os
from dotenv import load_dotenv
import json
from datetime import datetime

# Load environment variables
load_dotenv()

# --- 1. Use Your Existing TwoArmedBandit Class ---
class TwoArmedBandit:
    def __init__(self, epsilon=0.1, alpha=0.5):
        self.num_arms = 2  # Only two possible actions
        self.q_values = np.zeros(2)  # Two Q-values
        self.epsilon = epsilon  # Exploration probability
        self.alpha = alpha  # Learning rate

    def select_action(self):
        """Choose an action using epsilon-greedy strategy."""
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.num_arms)  # Explore
        return np.argmax(self.q_values)  # Exploit

    def update_q_values(self, action, reward):
        """Update Q-values using the RL update formula."""
        self.q_values[action] += self.alpha * (reward - self.q_values[action])

# --- 2. Simulate a Participant Using the Model ---
def simulate_trials(env_sequence, epsilon=0.1, alpha=0.5):
    model = TwoArmedBandit(epsilon=epsilon, alpha=alpha)
    history = []

    for prob_arm1, prob_arm2 in env_sequence:
        probs = [prob_arm1, prob_arm2]
        
        # Use the model's select_action method
        action = model.select_action()
        explored = action != np.argmax(model.q_values)
        
        reward = 1 if random.random() < probs[action] else 0
        
        # Use the model's update method
        model.update_q_values(action, reward)

        history.append({
            "q_values": [round(q, 3) for q in model.q_values],
            "action": action,
            "reward": reward,
            "explored": explored,
            "probs": probs.copy()
        })

    return history

# --- 3. Generate Environment Sequences ---
def get_environment_sequence(phases, trials_per_phase):
    return [env for env in phases for _ in range(trials_per_phase)]

# --- 4. Create Prompt Template ---
def generate_prompt(training_history, new_environment):
    """
    Generate prompt with history from training environment and ask for predictions in new environment
    
    Args:
        training_history: List of trials from the training environment
        new_environment: Tuple of (prob_arm1, prob_arm2) for the new environment
    """
    model_code = """
class TwoArmedBandit:
    def __init__(self, epsilon=0.1, alpha=0.5):
        self.num_arms = 2  # Only two possible actions
        self.q_values = np.zeros(2)  # Two Q-values
        self.epsilon = epsilon  # Exploration probability
        self.alpha = alpha  # Learning rate

    def select_action(self):
        \"\"\"Choose an action using epsilon-greedy strategy.\"\"\"
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.num_arms)  # Explore
        return np.argmax(self.q_values)  # Exploit

    def update_q_values(self, action, reward):
        \"\"\"Update Q-values using the RL update formula.\"\"\"
        self.q_values[action] += self.alpha * (reward - self.q_values[action])
"""

    prompt = f"""
You are an AI agent using Reinforcement Learning (RL) in a Two-Armed Bandit task. Your goal is to maximize total reward by balancing exploration (trying new actions) and exploitation (choosing the best-known action).

Below is the Python implementation of the Two-Armed Bandit model that follows Q-learning:

{model_code}

You have been learning in Training Environment with the following trial history:
"""

    for i, trial in enumerate(training_history):
        prompt += f"""
Trial {i + 1}:
- Chosen Arm: {trial['action'] + 1}
- Reward: {trial['reward']}
- Q-values: {trial['q_values']}
- Exploration: {trial['explored']}"""

    current_q = training_history[-1]['q_values'] if training_history else [0.0, 0.0]
    
    prompt += f"""

You are now placed in a new environment with different reward probabilities.
Your current Q-values from the training environment are: {current_q}

For each of the next 4 trials in this new environment, simulate the cognitive model's decision-making process step by step:

Trial 1:
1. Current Q-values: {current_q}
2. Using epsilon=0.1, would this be an exploration or exploitation trial? (Show your probability calculation)
3. Based on the above, which arm would you choose?
4. Assuming you receive a reward of 1, show how the Q-values would update
5. Assuming you receive a reward of 0, show how the Q-values would update

Trial 2:
1. Using the Q-values from Trial 1 (assume you got reward=1), make your next decision
2. Show the exploration vs exploitation calculation
3. Explain your arm choice
4. Show potential Q-value updates for both reward outcomes

Trial 3:
1. Using the Q-values from Trial 2 (assume you got reward=1), make your next decision
2. Show the exploration vs exploitation calculation
3. Explain your arm choice
4. Show potential Q-value updates for both reward outcomes

Trial 4:
1. Using the Q-values from Trial 3 (assume you got reward=1), make your next decision
2. Show the exploration vs exploitation calculation
3. Explain your arm choice
4. Show potential Q-value updates for both reward outcomes

For each trial, structure your response as:
1. Exploration/Exploitation Decision (with probability calculation)
2. Arm Choice (with explanation)
3. Q-value Updates:
   - If reward = 1: [show calculation]
   - If reward = 0: [show calculation]

Remember:
- You keep your initial Q-values from training: {current_q}
- Use epsilon-greedy with epsilon=0.1
- Learning rate alpha=0.5
- Show all calculations explicitly
"""

    return prompt

class OpenAIMultiModelPrompter:
    def __init__(self):
        # Initialize API key from environment variables
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        if not self.openai_api_key:
            raise ValueError("OpenAI API key not found in .env file")
        
        # Initialize the client
        self.client = OpenAI(api_key=self.openai_api_key)

        # Define available models
        self.models = {
            "gpt-4": "Most capable GPT-4 model",
            "gpt-4-turbo-preview": "Latest GPT-4 model, faster and cheaper than standard GPT-4",
            "gpt-3.5-turbo": "Fast and cost-effective model",
            "gpt-3.5-turbo-16k": "GPT-3.5 with longer context window",
        }

    def query_model(self, model: str, prompt: str) -> Dict:
        """Query specific OpenAI model"""
        try:
            # Updated API call syntax
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return {
                "model": model,
                "model_description": self.models.get(model, "No description available"),
                "prompt": prompt,
                "response": response.choices[0].message.content,
                "status": "success",
                "tokens_used": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            }
        except Exception as e:
            return {
                "model": model,
                "model_description": self.models.get(model, "No description available"),
                "prompt": prompt,
                "response": str(e),
                "status": "error",
                "tokens_used": None
            }

def run_model_comparison(training_env, test_env, trials_per_phase=4):
    """Run comparison across different OpenAI models"""
    prompter = OpenAIMultiModelPrompter()
    
    # Generate history from training environment
    env_sequence = get_environment_sequence([training_env], trials_per_phase)
    training_history = simulate_trials(env_sequence, epsilon=0.1, alpha=0.5)
    
    # Generate prompt with training history and new environment
    prompt = generate_prompt(training_history, test_env)
    
    results = []
    for model in prompter.models.keys():
        print(f"Querying {model}...")
        result = prompter.query_model(model, prompt)
        results.append(result)
    
    return training_history, results

def main():
    # Define training and test environment pairs
    environment_pairs = [
        {
            "training": (0.2, 0.2),    # Training: Both arms low
            "test": (0.8, 0.8),        # Test: Both arms high
            "name": "low_to_high1"
        },
        {
            "training": (0.3, 0.7),    # Training: Left low, right high
            "test": (0.7, 0.3),        # Test: Reversed probabilities
            "name": "probability_reversal1"
        },
        {
            "training": (0.8, 0.2),    # Training: Left high, right low
            "test": (0.2, 0.8),        # Test: Reversed probabilities
            "name": "extreme_reversal1"
        },
        {
            "training": (0.5, 0.5),    # Training: Equal probabilities
            "test": (0.9, 0.1),        # Test: Highly unequal
            "name": "equal_to_unequal1"
        }
    ]

    # Create timestamp for results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Process each environment pair
    for env_pair in environment_pairs:
        print(f"\n=== TRANSFER SCENARIO: {env_pair['name']} ===")
        
        # Run comparison across models
        training_history, results = run_model_comparison(
            env_pair["training"], 
            env_pair["test"]
        )
        
        # Generate the prompt for this scenario
        prompt = generate_prompt(training_history, env_pair["test"])
        
        # Save results to file with descriptive name
        output_file = f"bandit_transfer_{env_pair['name']}_{timestamp}.txt"
        
        with open(output_file, 'w') as f:
            # Write scenario details
            f.write(f"Transfer Learning Scenario: {env_pair['name']}\n")
            f.write("\nTraining Environment:\n")
            f.write(f"Arm1={env_pair['training'][0]}, Arm2={env_pair['training'][1]}\n")
            f.write("\nTest Environment:\n")
            f.write(f"Arm1={env_pair['test'][0]}, Arm2={env_pair['test'][1]}\n")
            
            # Write the training trial history
            f.write("\nTraining Environment History:\n")
            for trial_idx, trial in enumerate(training_history):
                f.write(f"Trial {trial_idx + 1}:\n")
                f.write(f"- Chosen Arm: {trial['action'] + 1}\n")
                f.write(f"- Reward: {trial['reward']}\n")
                f.write(f"- Q-values: {trial['q_values']}\n")
                f.write(f"- Exploration: {trial['explored']}\n")
                f.write(f"- True Probabilities: Arm1={trial['probs'][0]}, Arm2={trial['probs'][1]}\n")
            
            # Write the complete prompt being used
            f.write("\n" + "="*50 + "\n")
            f.write("COMPLETE PROMPT SENT TO MODELS:\n")
            f.write("="*50 + "\n")
            f.write(prompt)
            f.write("\n" + "="*50 + "\n")
            
            # Write model responses
            f.write("\nMODEL RESPONSES FOR TRANSFER LEARNING:\n")
            f.write("="*50 + "\n")
            for result in results:
                output = f"\n{'='*50}\n"
                output += f"Model: {result['model']}\n"
                output += f"Description: {result.get('model_description', 'No description available')}\n"
                output += f"Status: {result.get('status', 'unknown')}\n"
                
                tokens = result.get('tokens_used')
                if tokens:
                    output += f"Tokens Used: {tokens}\n"
                
                output += f"Response: {result.get('response', 'No response available')}\n"
                
                print(output)
                f.write(output)
        
        print(f"\nResults have been saved to {output_file}")

if __name__ == "__main__":
    main()
