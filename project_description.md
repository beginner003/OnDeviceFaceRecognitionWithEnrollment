Project 8: On-Device Continual Face Recognition with Forgetting 
Prevention 
Abstract 
This project builds an incremental face recognition system for home security cameras that 
continuously learns new identities without forgetting previously registered people. The system 
addresses a practical deployment scenario: a camera initially recognizes 3-5 family members, 
then incrementally learns new people (housekeepers, neighbors, frequent visitors) as they are 
registered by users over time. The core challenge is enabling on-device learning under strict 
resource constraints (8GB memory on raspberry pi 5) while preventing catastrophic forgetting—the 
tendency of neural networks to lose performance on old identities when trained on new ones. You 
will implement and compare continual learning techniques optimized for edge deployment, 
including exemplar-based replay, regularization methods, and architectural approaches. The 
system must support at least 10 people with incremental updates completing within 1 minute per 
new identity, maintain >95% accuracy on previously learned identities, and operate within the 
memory budget. Key evaluation metrics include average recognition accuracy across all learned 
identities, backward transfer (accuracy retention on old identities), memory footprint during 
training and inference, and incremental training time per new person. 
Expectations 
• Incremental learning pipeline: implement sequential learning of new identities with a realistic 
registration workflow (e.g., 5 family members, 1 housekeeper, 2 neighbors). Integrate at least 
two anti-forgetting techniques and compare their effectiveness. 
• Memory-efficient strategies: optimize for 8GB memory. Compare raw exemplar images vs. 
compressed feature vectors vs. synthetic replay. Implement intelligent sample selection to 
maximize retention within memory budget. Measure memory overhead of each approach. 
• Edge deployment and evaluation: deploy on raspberry and measure real-world 
performance. Test on face recognition datasets with incremental class learning protocol (split 
into multiple tasks). Report accuracy, forgetting rate, training time per new identity, and 
memory usage.  
• System demonstration: build a working prototype with user registration interface. Show 
concrete examples: register 5 people initially, add 3 more incrementally, verify recognition 
accuracy on all 8 people.  
Hardware may use 
• Rasberry pi 5 (8GB) ×1 
• intel real sense 
Software may use 
• PyTorch with TensorRT for deployment 
• Face recognition models: ArcFace, FaceNet, or MobileFaceNet 
• Continual learning: Avalanche library or custom implementation 
• Datasets: VGGFace2, MS-Celeb-1M (subsets)