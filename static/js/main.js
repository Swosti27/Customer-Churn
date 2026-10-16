document.addEventListener("DOMContentLoaded", () => {
    // Navigation Tabs
    const navSingle = document.getElementById("nav-single");
    const navBatch = document.getElementById("nav-batch");
    const navModel = document.getElementById("nav-model");

    const tabSingle = document.getElementById("tab-single");
    const tabBatch = document.getElementById("tab-batch");
    const tabModel = document.getElementById("tab-model");

    const navBtns = [navSingle, navBatch, navModel];
    const tabPanels = [tabSingle, tabBatch, tabModel];

    function switchTab(activeBtn, activePanel) {
        navBtns.forEach(b => b.classList.remove("active"));
        tabPanels.forEach(p => p.classList.add("hidden"));

        activeBtn.classList.add("active");
        activePanel.classList.remove("hidden");
    }

    if (navSingle) navSingle.addEventListener("click", (e) => { e.preventDefault(); switchTab(navSingle, tabSingle); });
    if (navBatch) navBatch.addEventListener("click", (e) => { e.preventDefault(); switchTab(navBatch, tabBatch); });
    if (navModel) navModel.addEventListener("click", (e) => { e.preventDefault(); switchTab(navModel, tabModel); });

    // Single Prediction Form Handler
    const singlePredictForm = document.getElementById("single-predict-form");
    const singleResultCard = document.getElementById("single-result-card");
    const riskPill = document.getElementById("risk-pill");
    const resultStatusTitle = document.getElementById("result-status-title");
    const resultScoreNum = document.getElementById("result-score-num");
    const barFill = document.getElementById("bar-fill");
    const recsUl = document.getElementById("recs-ul");
    const btnSingleSubmit = document.getElementById("btn-single-submit");

    if (singlePredictForm) {
        singlePredictForm.addEventListener("submit", async (e) => {
            e.preventDefault();

            btnSingleSubmit.disabled = true;
            btnSingleSubmit.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Evaluating...';

            const payload = {
                SeniorCitizen: document.getElementById("SeniorCitizen").checked ? 1 : 0,
                Partner: document.getElementById("Partner").checked ? "Yes" : "No",
                Dependents: document.getElementById("Dependents").checked ? "Yes" : "No",
                InternetService: document.getElementById("InternetService").value,
                MultipleLines: document.getElementById("MultipleLines").checked ? "Yes" : "No",
                OnlineSecurity: document.getElementById("OnlineSecurity").checked ? "Yes" : "No",
                OnlineBackup: document.getElementById("OnlineBackup").checked ? "Yes" : "No",
                DeviceProtection: document.getElementById("DeviceProtection").checked ? "Yes" : "No",
                TechSupport: document.getElementById("TechSupport").checked ? "Yes" : "No",
                StreamingTV: document.getElementById("StreamingTV").checked ? "Yes" : "No",
                StreamingMovies: document.getElementById("StreamingMovies").checked ? "Yes" : "No",
                PaperlessBilling: document.getElementById("PaperlessBilling").checked ? "Yes" : "No",
                Contract: document.getElementById("Contract").value,
                PaymentMethod: document.getElementById("PaymentMethod").value,
                tenure: document.getElementById("tenure").value,
                MonthlyCharges: document.getElementById("MonthlyCharges").value,
                TotalCharges: document.getElementById("TotalCharges").value
            };

            try {
                const response = await fetch("/predict", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(payload)
                });

                if (!response.ok) {
                    const err = await response.json();
                    alert(`Error: ${err.error || 'Failed to compute prediction'}`);
                    return;
                }

                const data = await response.json();
                renderSingleResult(data);

            } catch (err) {
                console.error(err);
                alert("Server error while evaluating risk.");
            } finally {
                btnSingleSubmit.disabled = false;
                btnSingleSubmit.innerHTML = '<i class="fa-solid fa-chart-line"></i> Evaluate Customer Risk';
            }
        });
    }

    function renderSingleResult(data) {
        singleResultCard.classList.remove("hidden");

        riskPill.textContent = `${data.risk_level} RISK`;
        riskPill.style.backgroundColor = `${data.risk_color}18`;
        riskPill.style.color = data.risk_color;
        riskPill.style.border = `1px solid ${data.risk_color}`;

        resultStatusTitle.textContent = data.prediction_label;
        resultScoreNum.textContent = `${data.churn_probability}%`;

        barFill.style.width = `${data.churn_probability}%`;
        barFill.style.backgroundColor = data.risk_color;

        recsUl.innerHTML = "";
        if (data.recommendations && data.recommendations.length > 0) {
            data.recommendations.forEach(rec => {
                const li = document.createElement("li");
                li.textContent = rec;
                recsUl.appendChild(li);
            });
        }

        singleResultCard.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }

    // Batch Upload Form Handler
    const batchFileElement = document.getElementById("batch-file-element");
    const batchFileName = document.getElementById("batch-file-name");
    const batchForm = document.getElementById("batch-form");
    const batchResultsBox = document.getElementById("batch-results-box");
    const btnBatchRun = document.getElementById("btn-batch-run");

    if (batchFileElement) {
        batchFileElement.addEventListener("change", (e) => {
            if (e.target.files && e.target.files.length > 0) {
                batchFileName.textContent = `Selected File: ${e.target.files[0].name}`;
            } else {
                batchFileName.textContent = "No file selected";
            }
        });
    }

    if (batchForm) {
        batchForm.addEventListener("submit", async (e) => {
            e.preventDefault();

            if (!batchFileElement.files || batchFileElement.files.length === 0) {
                alert("Please click to select a CSV file first.");
                return;
            }

            btnBatchRun.disabled = true;
            btnBatchRun.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Processing Predictions...';

            const formData = new FormData();
            formData.append("file", batchFileElement.files[0]);

            try {
                const response = await fetch("/predict_batch", {
                    method: "POST",
                    body: formData
                });

                if (!response.ok) {
                    const err = await response.json();
                    alert(`Batch Processing Error: ${err.error || 'Failed to process batch CSV'}`);
                    return;
                }

                const data = await response.json();
                if (data.success) {
                    renderBatchResults(data);
                } else {
                    alert("Failed to process batch predictions.");
                }

            } catch (err) {
                console.error(err);
                alert("Network error processing batch file.");
            } finally {
                btnBatchRun.disabled = false;
                btnBatchRun.innerHTML = '<i class="fa-solid fa-gears"></i> Run Batch Prediction';
            }
        });
    }

    function renderBatchResults(data) {
        batchResultsBox.classList.remove("hidden");

        const summary = data.summary;
        document.getElementById("batch-record-count").textContent = `Processed ${summary.total_records} customer records.`;
        document.getElementById("m-total").textContent = summary.total_records;
        document.getElementById("m-churn").textContent = summary.predicted_churn;
        document.getElementById("m-retain").textContent = summary.predicted_retain;
        document.getElementById("m-highrisk").textContent = summary.high_risk_count;

        const downloadLink = document.getElementById("download-csv-link");
        downloadLink.href = data.download_url;

        // Render Table Headers & Rows
        const thRow = document.getElementById("th-row");
        const tbRows = document.getElementById("tb-rows");

        thRow.innerHTML = "";
        tbRows.innerHTML = "";

        if (!data.preview || data.preview.length === 0) return;

        const displayCols = ['customerID', 'tenure', 'Contract', 'MonthlyCharges', 'TotalCharges', 'Churn_Probability_%', 'Predicted_Label', 'Risk_Level'];
        const colsToRender = displayCols.filter(col => data.columns.includes(col));

        if (colsToRender.length < 3) {
            colsToRender.length = 0;
            colsToRender.push(...data.columns.slice(0, 8));
        }

        // Header Row
        colsToRender.forEach(col => {
            const th = document.createElement("th");
            th.textContent = col;
            thRow.appendChild(th);
        });

        // Body Rows
        data.preview.forEach(row => {
            const tr = document.createElement("tr");
            colsToRender.forEach(col => {
                const td = document.createElement("td");
                let val = row[col];
                if (col === 'Risk_Level') {
                    const badge = document.createElement("span");
                    badge.className = "status-pill";
                    badge.textContent = val;
                    if (val === 'HIGH') {
                        badge.style.color = '#dc2626'; badge.style.backgroundColor = '#fee2e2';
                    } else if (val === 'MEDIUM') {
                        badge.style.color = '#d97706'; badge.style.backgroundColor = '#fef3c7';
                    } else {
                        badge.style.color = '#16a34a'; badge.style.backgroundColor = '#dcfce7';
                    }
                    td.appendChild(badge);
                } else if (col === 'Churn_Probability_%') {
                    td.innerHTML = `<strong>${val}%</strong>`;
                } else {
                    td.textContent = val !== undefined ? val : 'N/A';
                }
                tr.appendChild(td);
            });
            tbRows.appendChild(tr);
        });

        batchResultsBox.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }
});
