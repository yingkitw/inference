fn main() {
    println!("Influence CLI - Usage Examples\n");
    
    println!("1. Set environment variables:");
    println!("   export WATSONX_API_KEY=\"your-api-key-here\"");
    println!("   export WATSONX_PROJECT_ID=\"your-project-id-here\"");
    println!("   export WATSONX_MODEL_ID=\"ibm/granite-4-h-small\"");
    println!();
    
    println!("2. Download model from HuggingFace mirror:");
    println!("   cargo run -- download -m ibm/granite-4-h-small");
    println!();
    
    println!("3. Generate text with a prompt:");
    println!("   cargo run -- generate \"What is Rust programming language?\"");
    println!();
    
    println!("4. Serve the influencer:");
    println!("   cargo run -- serve --port 8080");
    println!();
    
    println!("For more information, see README.md");
}
