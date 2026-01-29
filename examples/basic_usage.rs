use std::env;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    env::set_var("WATSONX_API_KEY", "your-api-key-here");
    env::set_var("WATSONX_PROJECT_ID", "your-project-id-here");
    env::set_var("WATSONX_MODEL_ID", "ibm/granite-4-h-small");

    println!("Example: Download model from HuggingFace mirror");
    println!("Run: cargo run -- download -m ibm/granite-4-h-small");
    println!();
    
    println!("Example: Generate text with a prompt");
    println!("Run: cargo run -- generate \"What is Rust programming language?\"");
    println!();
    
    println!("Example: Serve the influencer");
    println!("Run: cargo run -- serve --port 8080");
    
    Ok(())
}
