# API Key Management

This project uses `.env` files to manage API keys securely. This approach allows you to:

- Keep API keys out of your code
- Avoid hardcoding sensitive information
- Share configuration without exposing credentials
- Use different keys for different environments

## Setup

### 1. Create `.env` File

Copy the example file and add your actual API keys:

```bash
cp .env.example .env
```

### 2. Edit `.env` File

Open `.env` in your editor and add your API keys:

```bash
# DashScope API Key for Qwen models
DASHSCOPE_API_KEY=your-actual-dashscope-api-key-here

# OpenAI API Key (optional)
OPENAI_API_KEY=your-actual-openai-api-key-here
```

### 3. Security

The `.env` file is already included in `.gitignore`, so it won't be committed to version control.

**Never commit `.env` files with real API keys!**

## Usage

### Automatic Loading

All evaluation scripts automatically load environment variables from `.env`:

```bash
# Just run the script - it will automatically load API keys from .env
bash scripts/gsm8k_qwen3_32b_api.sh
```

### Manual Override

You can still override environment variables if needed:

```bash
# Override API key for a single run
DASHSCOPE_API_KEY=another-key bash scripts/gsm8k_qwen3_32b_api.sh

# Override model name
MODEL_NAME=gpt-4 bash scripts/gsm8k_qwen3_32b_api.sh

# Override sample count
NUM_SAMPLES=50 bash scripts/gsm8k_qwen3_32b_api.sh
```

## Supported API Keys

### DashScope API Key

Used for Qwen models via Alibaba Cloud's DashScope service.

```bash
DASHSCOPE_API_KEY=your-dashscope-key
```

Get your key from: https://dashscope.console.aliyun.com/

### OpenAI API Key

Used for OpenAI models (GPT-3.5, GPT-4, etc.).

```bash
OPENAI_API_KEY=your-openai-key
```

Get your key from: https://platform.openai.com/api-keys

## Adding New API Keys

To add support for a new API key:

1. Add it to `.env.example`:

```bash
# Your New API Provider
NEW_API_KEY=your-new-api-key-here
```

2. Update the relevant configuration or script to read it:

```python
api_key = os.getenv("NEW_API_KEY")
```

3. Update `.gitignore` to ensure `.env` is never committed (already done).

## Best Practices

1. **Never commit `.env` files** - They're already in `.gitignore`
2. **Use different keys for development and production**
3. **Rotate keys regularly** for security
4. **Use environment-specific `.env` files** (e.g., `.env.dev`, `.env.prod`)
5. **Document required keys** in `.env.example`

## Troubleshooting

### API Key Not Found

If you see an error about missing API keys:

```bash
Error: DASHSCOPE_API_KEY environment variable is not set
```

**Solution**: Make sure you've created `.env` and added your API key.

### Wrong API Key

If authentication fails:

```bash
Error: Authentication failed
```

**Solution**: Verify your API key is correct in `.env` file.

### Key Not Loading

If scripts don't seem to be using your `.env` file:

1. Check that `.env` is in the project root directory
2. Verify the file format (no spaces around `=`)
3. Make sure you're running scripts from the project root

## Example `.env` File

```bash
# API Keys Configuration
# Copy this file to .env and fill in your actual API keys

# DashScope API Key for Qwen models
DASHSCOPE_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# OpenAI API Key (optional)
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Other API keys can be added here as needed
```
