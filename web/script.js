document.addEventListener("DOMContentLoaded", () => {
    const uploadZone = document.getElementById("upload-zone");
    const fileInput = document.getElementById("file-input");
    const uploadContent = document.querySelector(".upload-content");
    const previewContainer = document.getElementById("preview-container");
    const imagePreview = document.getElementById("image-preview");
    const videoPreview = document.getElementById("video-preview");
    const analyzeBtn = document.getElementById("analyze-btn");
    const removeBtn = document.getElementById("remove-btn");

    const loader = document.getElementById("loader");
    const resultsZone = document.getElementById("results-zone");
    const restartBtn = document.getElementById("restart-btn");

    let currentFile = null;

    // Trigger file select dialog
    uploadZone.addEventListener('click', (e) => {
        // Prevent click if we click analyze/remove buttons or if image is loaded
        if (e.target.tagName !== "BUTTON" && !currentFile) {
            fileInput.click();
        }
    });

    // Drag-and-drop Events
    uploadZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        if(!currentFile) uploadZone.classList.add("dragover");
    });

    uploadZone.addEventListener('dragleave', () => {
        uploadZone.classList.remove("dragover");
    });

    uploadZone.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadZone.classList.remove("dragover");
        if (!currentFile && e.dataTransfer.files.length) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener('change', () => {
        if (fileInput.files.length) {
            handleFile(fileInput.files[0]);
        }
    });

    function handleFile(file) {
        currentFile = file;
        const fileURL = URL.createObjectURL(file);
        
        if (file.type.startsWith("video/")) {
            imagePreview.style.display = "none";
            videoPreview.src = fileURL;
            videoPreview.style.display = "block";
        } else {
            videoPreview.style.display = "none";
            imagePreview.src = fileURL;
            imagePreview.style.display = "block";
        }

        uploadContent.style.display = "none";
        previewContainer.style.display = "flex";
        uploadZone.style.borderStyle = "solid";
        uploadZone.style.padding = "2rem";
    }

    removeBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        resetUI();
    });

    analyzeBtn.addEventListener('click', async (e) => {
        e.stopPropagation(); 
        if (!currentFile) return;

        // Transition to loader
        uploadZone.classList.add("hidden");
        loader.classList.remove("hidden");

        // Cycle text to indicate work is happening
        const steps = document.querySelectorAll('.step');
        let currentStep = 0;
        const interval = setInterval(() => {
            if(currentStep < steps.length - 1) {
                steps[currentStep].classList.remove('active');
                currentStep++;
                steps[currentStep].classList.add('active');
            }
        }, 1500);

        try {
            const formData = new FormData();
            formData.append('file', currentFile);
            
            const response = await fetch('/api/analyze', {
                method: 'POST',
                body: formData
            });

            clearInterval(interval);
            
            if (!response.ok) {
                const errData = await response.json();
                throw new Error(errData.error || "Server format error");
            }
            
            // Read SSE stream
            const reader = response.body.getReader();
            const decoder = new TextDecoder('utf-8');
            let buffer = '';
            
            // Set up UI for streaming
            loader.classList.add("hidden");
            resultsZone.classList.remove("hidden");
            
            const toolsGrid = document.getElementById("tools-grid");
            toolsGrid.innerHTML = '';
            
            const renderToolName = (name) => {
                return name.replace('run_', '').replace('check_', '').replace(/_/g, ' ');
            };
            
            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                
                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n\n');
                
                for (let i = 0; i < lines.length - 1; i++) {
                    const line = lines[i].trim();
                    if (!line.startsWith('data: ')) continue;
                    
                    const payload = JSON.parse(line.substring(6));
                    
                    if (payload.event_type === 'init') {
                        document.getElementById("faces-detected").textContent = payload.faces_detected;
                    } 
                    else if (payload.event_type === 'tool_complete') {
                        // Create and append tool card dynamically
                        const res = payload.data;
                        const toolName = renderToolName(payload.tool_name);
                        
                        let statusClass = "status-error";
                        let statusText = "ERROR";

                        if (res.success) {
                            const realProb = 1.0 - res.score;
                            const isRisk = realProb < 0.45;
                            statusClass = isRisk ? "status-invalid" : "status-valid";
                            statusText = isRisk ? "SUSPICIOUS" : "CLEAR";
                        }

                        const card = document.createElement('div');
                        card.className = "tool-card";
                        
                        let cardInner = `
                            <div class="tool-header">
                                <div class="tool-name">${toolName.toUpperCase()}</div>
                                <div class="tool-status ${statusClass}">${statusText}</div>
                            </div>`;
                        
                        if (res.success) {
                            const realProb = 1.0 - res.score;
                            cardInner += `
                            <div class="tool-metrics">
                                <div class="metric">
                                    <span class="metric-label">Authenticity</span>
                                    <span class="metric-value" style="color: ${realProb < 0.5 ? 'var(--alert)' : 'var(--success)'}">${(realProb * 100).toFixed(0)}%</span>
                                </div>
                                <div class="metric">
                                    <span class="metric-label">Confidence</span>
                                    <span class="metric-value">${(res.confidence || 0).toFixed(2)}</span>
                                </div>
                            </div>`;
                        }

                        cardInner += `
                            <div class="tool-evidence" style="margin-top: 10px;">
                                ${res.success ? res.evidence_summary : (res.error_msg || "Tool execution failed.")}
                            </div>
                        `;
                        
                        card.innerHTML = cardInner;
                        toolsGrid.appendChild(card);
                    }
                    else if (payload.event_type === 'verdict') {
                        // Show final verdict and LLM explanation
                        const finalDoc = payload.data;
                        const finalVerdictBanner = document.getElementById("final-verdict-banner");
                        const finalVerdictText = document.getElementById("final-verdict");
                        const finalScoreText = document.getElementById("final-score");
                        const explanationBox = document.getElementById("explanation-box");
                        const explanationText = document.getElementById("explanation-text");
                        
                        const scorePercent = ((finalDoc.score || 0) * 100).toFixed(1);
                        finalScoreText.textContent = `${scorePercent}%`;
                        
                        if (finalDoc.explanation) {
                            explanationBox.style.display = "block";
                            explanationText.textContent = finalDoc.explanation;
                        }
                        
                        finalVerdictBanner.classList.remove("fake", "real");
                        if (finalDoc.verdict === "FAKE") {
                            finalVerdictText.textContent = "⚠️ FAKE MEDIA DETECTED";
                            finalVerdictBanner.classList.add("fake");
                        } else {
                            finalVerdictText.textContent = "✅ AUTHENTIC MEDIA";
                            finalVerdictBanner.classList.add("real");
                        }
                    }
                    else if (payload.event_type === 'early_stop') {
                        // Display early stop banner
                        const card = document.createElement('div');
                        card.className = "tool-card";
                        card.style.borderColor = "var(--primary-color)";
                        card.innerHTML = `
                            <div class="tool-header">
                                <div class="tool-name">PIPELINE EARLY STOP</div>
                                <div class="tool-status status-valid">HALTED</div>
                            </div>
                            <div class="tool-evidence" style="margin-top: 10px;">
                                Reason: ${payload.data.reason}. Confidence Lock-in: ${(payload.data.confidence).toFixed(2)}
                            </div>
                        `;
                        toolsGrid.appendChild(card);
                    }
                }
                buffer = lines[lines.length - 1]; // keep remaining buffer
            }

        } catch (error) {
            clearInterval(interval);
            alert("Error: " + error.message);
            resetUI();
        }
    });

    function resetUI() {
        if (imagePreview.src) {
            URL.revokeObjectURL(imagePreview.src);
            imagePreview.src = "";
        }
        if (videoPreview.src) {
            URL.revokeObjectURL(videoPreview.src);
            videoPreview.src = "";
        }

        currentFile = null;
        fileInput.value = "";
        uploadContent.style.display = "block";
        previewContainer.style.display = "none";
        
        videoPreview.style.display = "none";
        imagePreview.style.display = "none";

        uploadZone.classList.remove("hidden");
        loader.classList.add("hidden");
        resultsZone.classList.add("hidden");
        
        uploadZone.style.borderStyle = "dashed";
        uploadZone.style.padding = "3rem 2rem";
        
        // Reset active steps
        const steps = document.querySelectorAll('.step');
        steps.forEach(s => s.classList.remove('active'));
        steps[0].classList.add('active');
    }

    restartBtn.addEventListener('click', resetUI);
});
