document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('uploadForm');
    const frontInput = document.getElementById('front');
    const backInput = document.getElementById('back');
    const errorMessage = document.getElementById('errorMessage');
    const frontPreview = document.getElementById("frontPreview");
    const backPreview = document.getElementById("backPreview");
    const frontDropZone = document.getElementById("frontImageDropZone");
    const backDropZone = document.getElementById("backImageDropZone");

    // Function to validate file type
    function validateFileType(fileInput) {
        const file = fileInput.files[0];
        return file && file.type.startsWith('image/');
    }

    // Function to display error messages
    function displayError(message) {
        errorMessage.textContent = message;
        errorMessage.style.display = 'block';
    }

    // Function to preview image
    function previewImage(input, previewElement) {
        const file = input.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = function (e) {
                previewElement.src = e.target.result;
                previewElement.style.display = "block";
            };
            reader.readAsDataURL(file);
        }
    }

    // Event listener for form submission
    form.addEventListener('submit', (event) => {
        errorMessage.style.display = 'none';

        if (!frontInput.files.length || !backInput.files.length) {
            displayError('Please upload both front and back images.');
            event.preventDefault();
            return;
        }

        if (!validateFileType(frontInput) || !validateFileType(backInput)) {
            displayError('Only image files are allowed.');
            event.preventDefault();
        }
    });

    // Function to add drag-and-drop functionality
    function addDragAndDropHandlers(dropZone, input, previewElement) {
        dropZone.addEventListener("dragover", (e) => {
            e.preventDefault();
            dropZone.classList.add("drag-over");
        });

        dropZone.addEventListener("dragleave", () => {
            dropZone.classList.remove("drag-over");
        });

        dropZone.addEventListener("drop", (e) => {
            e.preventDefault();
            dropZone.classList.remove("drag-over");
            const files = e.dataTransfer.files;

            if (files.length > 0) {
                // Create a new DataTransfer to assign files
                const dataTransfer = new DataTransfer();
                dataTransfer.items.add(files[0]); // Add the dropped file to DataTransfer
                input.files = dataTransfer.files; // Assign to input
                previewImage(input, previewElement);
            }
        });

        // Click to upload
        dropZone.addEventListener("click", () => input.click());
    }

    // Apply drag-and-drop handlers to both image zones
    addDragAndDropHandlers(frontDropZone, frontInput, frontPreview);
    addDragAndDropHandlers(backDropZone, backInput, backPreview);
});
