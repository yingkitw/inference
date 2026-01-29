use syntect::easy::HighlightLines;
use syntect::highlighting::{Style, ThemeSet};
use syntect::parsing::SyntaxSet;
use syntect::util::{as_24_bit_terminal_escaped, LinesWithEndings};
use termimad::{MadSkin, crossterm::style::Color};

pub struct OutputFormatter {
    skin: MadSkin,
    syntax_set: SyntaxSet,
    theme_set: ThemeSet,
}

impl OutputFormatter {
    pub fn new() -> Self {
        let mut skin = MadSkin::default();
        
        skin.bold.set_fg(Color::Cyan);
        skin.italic.set_fg(Color::Yellow);
        skin.headers[0].set_fg(Color::Green);
        skin.headers[1].set_fg(Color::Blue);
        skin.headers[2].set_fg(Color::Magenta);
        skin.code_block.set_fg(Color::White);
        skin.inline_code.set_fg(Color::Yellow);
        
        let syntax_set = SyntaxSet::load_defaults_newlines();
        let theme_set = ThemeSet::load_defaults();
        
        Self {
            skin,
            syntax_set,
            theme_set,
        }
    }
    
    pub fn print_markdown(&self, text: &str) {
        // Parse and handle code blocks with syntax highlighting
        let mut current_pos = 0;
        let text_bytes = text.as_bytes();
        
        while current_pos < text.len() {
            // Look for code block start
            if let Some(code_start) = text[current_pos..].find("```") {
                let absolute_start = current_pos + code_start;
                
                // Print text before code block
                if code_start > 0 {
                    let before_text = &text[current_pos..absolute_start];
                    self.skin.print_text(before_text);
                }
                
                // Find language identifier (first line after ```)
                let after_marker = absolute_start + 3;
                let line_end = text[after_marker..].find('\n').unwrap_or(0);
                let lang_line = &text[after_marker..after_marker + line_end].trim();
                
                // Find code block end
                if let Some(code_end_offset) = text[after_marker + line_end..].find("```") {
                    let code_end = after_marker + line_end + code_end_offset;
                    let code_content = &text[after_marker + line_end + 1..code_end];
                    
                    // Print code block with syntax highlighting
                    println!();
                    self.print_code(code_content, lang_line);
                    
                    // Move past the closing ```
                    current_pos = code_end + 3;
                    
                    // Skip newline after closing ```
                    if current_pos < text.len() && text_bytes[current_pos] == b'\n' {
                        current_pos += 1;
                    }
                } else {
                    // No closing ```, treat as regular text
                    self.skin.print_text(&text[current_pos..]);
                    break;
                }
            } else {
                // No more code blocks, print remaining text
                self.skin.print_text(&text[current_pos..]);
                break;
            }
        }
    }
    
    pub fn print_code(&self, code: &str, language: &str) {
        let syntax = self.syntax_set
            .find_syntax_by_extension(language)
            .or_else(|| self.syntax_set.find_syntax_by_name(language))
            .unwrap_or_else(|| self.syntax_set.find_syntax_plain_text());
        
        let theme = &self.theme_set.themes["base16-ocean.dark"];
        let mut highlighter = HighlightLines::new(syntax, theme);
        
        // Print language tag if available
        if !language.is_empty() {
            println!("┌─ {} ─", language);
        } else {
            println!("┌─ code ─");
        }
        
        for line in LinesWithEndings::from(code) {
            let ranges: Vec<(Style, &str)> = highlighter
                .highlight_line(line, &self.syntax_set)
                .unwrap_or_default();
            let escaped = as_24_bit_terminal_escaped(&ranges[..], false);
            print!("│ {}", escaped);
        }
        println!("└─────");
    }
    
    pub fn print_header(&self, text: &str) {
        self.print_markdown(&format!("# {}\n", text));
    }
    
    pub fn print_section(&self, title: &str, content: &str) {
        self.print_markdown(&format!("## {}\n\n{}\n", title, content));
    }
    
    pub fn print_info(&self, text: &str) {
        self.print_markdown(&format!("**ℹ️  {}**\n", text));
    }
    
    pub fn print_success(&self, text: &str) {
        self.print_markdown(&format!("**✓ {}**\n", text));
    }
    
    pub fn print_warning(&self, text: &str) {
        self.print_markdown(&format!("**⚠️  {}**\n", text));
    }
    
    pub fn print_error(&self, text: &str) {
        self.print_markdown(&format!("**❌ {}**\n", text));
    }
    
    pub fn print_list_item(&self, text: &str) {
        self.print_markdown(&format!("- {}\n", text));
    }
    
    pub fn print_model_info(&self, name: &str, path: &str, format: &str, arch: &str, size: &str, files: usize) {
        let info = format!(
r#"### {}

- **Path:** `{}`
- **Format:** {}
- **Architecture:** {}
- **Size:** {} ({} files)
"#,
            name, path, format, arch, size, files
        );
        self.print_markdown(&info);
    }
    
    pub fn print_search_result(&self, idx: usize, model_id: &str, author: Option<&str>, task: Option<&str>, downloads: Option<u64>, likes: Option<u64>, library: Option<&str>) {
        let mut result = format!("### {}. {}\n\n", idx, model_id);
        
        if let Some(a) = author {
            result.push_str(&format!("- **Author:** {}\n", a));
        }
        if let Some(t) = task {
            result.push_str(&format!("- **Task:** {}\n", t));
        }
        if let Some(d) = downloads {
            result.push_str(&format!("- **Downloads:** {}\n", d));
        }
        if let Some(l) = likes {
            result.push_str(&format!("- **Likes:** {}\n", l));
        }
        if let Some(lib) = library {
            result.push_str(&format!("- **Library:** {}\n", lib));
        }
        
        result.push_str(&format!("\n**Download:** `influence download -m {}`\n", model_id));
        
        self.print_markdown(&result);
    }
    
    pub fn print_chat_header(&self) {
        let header = r#"
# Interactive Chat Mode

**Commands:**
- Type your messages and press Enter
- `/help` - Show available commands
- `/quit` or `/exit` - Exit chat mode
"#;
        self.print_markdown(header);
    }
    
    pub fn print_help_commands(&self) {
        let help = r#"
## Chat Commands

- `/help` - Show this help message
- `/clear` - Clear conversation history
- `/history` - Show conversation history
- `/save <filename>` - Save conversation to file
- `/load <filename>` - Load conversation from file
- `/set <param> <val>` - Change parameters (temperature, top_p)
- `/quit` or `/exit` - Exit chat mode

**Example:** `/set temperature 0.8`
"#;
        self.print_markdown(help);
    }
}

impl Default for OutputFormatter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_output_formatter_creation() {
        let formatter = OutputFormatter::new();
        assert!(formatter.syntax_set.syntaxes().len() > 0);
        assert!(formatter.theme_set.themes.len() > 0);
    }
}
