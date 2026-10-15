// DeepfakeDetector — Upload page interactivity
// Handles drag-and-drop, file selection preview, and a client-side
// "processing" progress indicator while the server runs the detection
// pipeline (frame extraction -> face detection -> CNN-LSTM / heuristic
// inference).

document.addEventListener("DOMContentLoaded", function () {
  const uploadBox = document.getElementById("upload-box");
  const fileInput = document.getElementById("video-input");
  const fileNameLabel = document.getElementById("file-name");
  const form = document.getElementById("upload-form");
  const progressWrapper = document.getElementById("progress-wrapper");
  const submitBtn = document.getElementById("submit-btn");

  if (!uploadBox) return;

  uploadBox.addEventListener("click", () => fileInput.click());

  ["dragenter", "dragover"].forEach((evt) => {
    uploadBox.addEventListener(evt, (e) => {
      e.preventDefault();
      uploadBox.classList.add("dragover");
    });
  });

  ["dragleave", "drop"].forEach((evt) => {
    uploadBox.addEventListener(evt, (e) => {
      e.preventDefault();
      uploadBox.classList.remove("dragover");
    });
  });

  uploadBox.addEventListener("drop", (e) => {
    const files = e.dataTransfer.files;
    if (files.length) {
      fileInput.files = files;
      updateFileName();
    }
  });

  fileInput.addEventListener("change", updateFileName);

  function updateFileName() {
    if (fileInput.files.length) {
      const file = fileInput.files[0];
      const sizeMB = (file.size / (1024 * 1024)).toFixed(2);
      fileNameLabel.textContent = `${file.name} (${sizeMB} MB)`;
    }
  }

  if (form) {
    form.addEventListener("submit", () => {
      if (!fileInput.files.length) return; // let native validation handle it
      progressWrapper.style.display = "block";
      submitBtn.disabled = true;
      submitBtn.textContent = "Analyzing video...";
    });
  }
});
