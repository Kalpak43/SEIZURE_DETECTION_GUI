// Get the upload button and file input element
const fileUploadInput = document.getElementById("file-upload");
const fileUploadButton = document.getElementById("file-upload-btn");

let filePath = null;
let progress = 0;

const plotButton = document.getElementById("plot-btn");
const processButton = document.getElementById("process-btn");
const plotWarning = document.getElementById("plot-warning");
const processWarning = document.getElementById("process-warning");
const results = document.getElementById("predictions");

if (!filePath) {
  plotButton.style.display = "none";
  plotWarning.style.display = "block";
  processButton.style.display = "none";
  processWarning.style.display = "block";
  results.style.display = "none";
}

// Handle file upload
fileUploadButton.addEventListener("click", async () => {
  const file = fileUploadInput.files[0];

  if (file) {
    // Prepare the form data
    const formData = new FormData();
    formData.append("file", file);

    // Send POST request to upload the file
    const response = await fetch("/upload", {
      method: "POST",
      body: formData,
    });

    if (response.ok) {
      const responseData = await response.json();
      filePath = responseData.file_path; // Get the file path from the response

      if (filePath) {
        // Enable the plot and process buttons
        plotButton.style.display = "inline-block";
        plotWarning.style.display = "none";
        processButton.style.display = "inline-block";
        processWarning.style.display = "none";
      }
    } else {
      console.error("Error with file processing request:", response.statusText);
    }
  } else {
    alert("Please select a file to upload.");
  }
});

plotButton.addEventListener("click", async () => {
  try {
    processButton.disabled = true;
    const loader = document.createElement("div");
    loader.className = "loader";

    document.getElementById("plot").innerHTML = "";
    document.getElementById("plot").appendChild(loader);

    // Send POST request to /plot_edf route
    const response = await fetch("/plot_edf", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ file_path: filePath }),
    });

    // Parse JSON response
    const data = await response.json();

    if (response.ok && data.image) {
      // Display the image by setting src of img tag
      const img = document.createElement("img");
      img.id = "plot-image";
      img.src = "data:image/png;base64," + data.image;
      img.width = 500;
      img.height = 500;
      document.getElementById("plot").innerHTML = "";
      document.getElementById("plot").appendChild(img);
    } else {
      console.error("Error:", data.error || "Could not generate plot.");
      alert(data.error || "Failed to generate plot.");
    }

    processButton.disabled = false;
  } catch (error) {
    console.error("Error:", error);
    alert("Error generating plot.");
  }
});
processButton.addEventListener("click", async () => {
  try {

    document.getElementById("predict").style.justifyContent = 'start'

    processButton.style.display = "none";
    results.style.display = "block";

    // Start streaming predictions
    const eventSource = new EventSource(
      `/stream_predictions?file_path=${encodeURIComponent(filePath)}`
    );


    eventSource.onmessage = function (event) {
      const data = JSON.parse(event.data);
      console.log("Data", data);

      // Update the predictions progress
      const progress = data.progress;
      const progressBar = document.getElementById("progress-bar");
      progressBar.style.width = progress + "%";

      // const progressText = document.getElementById("progress-text");
      // if (progress < 100) {
      //   progressText.innerHTML = `Progress: ${progress}%`;
      // } else if (progress === 100) {
      //   progressText.innerHTML = "Processing complete!";
      //   eventSource.close(); // Close the connection after processing is complete
      // }

      // Populate Seizure Events Table
      const seizureTableBody = document
        .getElementById("seizure-events-table")
        .querySelector("tbody");

      // Clear previous entries
      seizureTableBody.innerHTML = "";

      data.seizure_events.forEach((event) => {
        const row = document.createElement("tr");
        row.innerHTML = `
    <td>${event.start}</td>
    <td>${event.end}</td>
    <td>${event.prediction_class}</td>
  `;
        seizureTableBody.appendChild(row);
      });

      // Populate Non-Seizure Events Table
      const nonSeizureTableBody = document
        .getElementById("non-seizure-events-table")
        .querySelector("tbody");

      // Clear previous entries
      nonSeizureTableBody.innerHTML = "";

      data.non_seizure_events.forEach((event) => {
        const row = document.createElement("tr");
        row.innerHTML = `
    <td>${event.start}</td>
    <td>${event.end}</td>
  `;
        nonSeizureTableBody.appendChild(row);
      });
    };

    eventSource.onerror = function (error) {
      console.error("Error in receiving SSE:", error);
      eventSource.close();
    };
  } catch (e) {
    console.error("Error:", e);
    alert("Error processing the file.");
  }
});
