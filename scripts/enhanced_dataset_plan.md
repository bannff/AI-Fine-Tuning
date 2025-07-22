# Enhanced Cybersecurity Training Dataset Plan
# ...
# Goals:
# 1. Fix `\n` formatting issues in responses
# 2. Add high-quality cybersecurity datasets (<10GB each)
# ...

# (rest of the existing content)

# Move this file to scripts/
# Enhanced Cybersecurity Training Dataset Plan

## 🎯 Goals:
1. Fix `\n` formatting issues in responses
2. Add high-quality cybersecurity datasets (<10GB each)
3. Increase LoRA alpha influence (now set to 64)
4. Improve model's specialized cybersecurity knowledge

## 📊 High-Quality Datasets to Add (All <10GB):

### **Tier 1: Premium Quality**
1. **`moyix/SecurityEval`** - Academic security evaluation dataset
2. **`Vanessasml/cybersecurity_32k_instruction_input_output`** - 32k instruction tuning
3. **`Nitral-AI/Cybersecurity-ShareGPT`** - Conversational cybersecurity data
4. **`dattaraj/security-attacks-MITRE`** - MITRE ATT&CK framework data
5. **`CyberNative/CyberSecurityEval`** - Comprehensive evaluation suite

### **Tier 2: Specialized Knowledge**
1. **`vinitvek/cybersecurityattacks`** - Attack pattern datasets
2. **`cw1521/ember2018-malware`** - Malware analysis dataset
3. **`bnsapa/cybersecurity-ner`** - Named entity recognition for cybersecurity
4. **`schooly/Cyber-Security-Breaches`** - Incident response cases
5. **`jcordon5/cybersecurity-rules`** - Security policy rules

### **Tier 3: Code Security**
1. **`zeroshot/cybersecurity-corpus`** - General cybersecurity corpus
2. **`AR2021/cybersecurity-corpus-llama2-1k`** - LLaMA-optimized dataset
3. **`Falah/Practical_AI_for_Cybersecurity`** - Practical AI applications

## 🔧 Enhanced Training Configuration:

### LoRA Parameters (Updated):
```yaml
lora_parameters:
  rank: 16
  alpha: 64      # Doubled for stronger influence
  dropout: 0.1
  scale: 3.0     # Increased scaling
```

### Data Processing Priority:
1. **Fix newline issues**: Clean datasets to prevent `\n` artifacts
2. **Format consistency**: Ensure conversational format
3. **Quality filtering**: Remove duplicate/low-quality samples
4. **STRIDE focus**: Emphasize correct STRIDE definitions

## 📈 Expected Improvements:
- ✅ Stronger adapter influence (alpha: 32→64)
- ✅ Better formatting (no more `\n` artifacts)
- ✅ More specialized knowledge
- ✅ Improved technical accuracy
- ✅ Better STRIDE understanding

## 🚀 Next Steps:
1. Download and process Tier 1 datasets
2. Clean formatting issues
3. Retrain with enhanced configuration
4. Test with technical cybersecurity questions
