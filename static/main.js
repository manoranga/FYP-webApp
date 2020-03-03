//========================================================================
// Drag and drop image handling
//========================================================================

var fileDrag = document.getElementById("file-drag");
var fileSelect = document.getElementById("file-upload");

// Add event listeners
fileDrag.addEventListener("dragover", fileDragHover, false);
fileDrag.addEventListener("dragleave", fileDragHover, false);
fileDrag.addEventListener("drop", fileSelectHandler, false);
fileSelect.addEventListener("change", fileSelectHandler, false);

function fileDragHover(e) {
  // prevent default behaviour
  e.preventDefault();
  e.stopPropagation();

  fileDrag.className = e.type === "dragover" ? "upload-box dragover" : "upload-box";
}

function fileSelectHandler(e) {
  // handle file selecting
  var files = e.target.files || e.dataTransfer.files;
  fileDragHover(e);
  for (var i = 0, f; (f = files[i]); i++) {
    previewFile(f);
  }
}

//========================================================================
// Web page elements for functions to use
//========================================================================

var imagePreview = document.getElementById("image-preview");
var imageDisplay = document.getElementById("image-display");
var uploadCaption = document.getElementById("upload-caption");
var predResult = document.getElementById("pred-result");
var predprobability = document.getElementById("predict-probability");
var predResult_segmeted_NoRotated = document.getElementById("pred-result-segmented");
var predprobability_segmeted_NoRotated = document.getElementById("predict-probability-segmented");
var imageDisplay_segmented_NoRotated = document.getElementById("image-display_segmented");
var predResult_rotated = document.getElementById("pred-result_rotated");
var predprobability_rotated = document.getElementById("predict-probability_rotated");
var imageDisplay_rotated = document.getElementById("image-display_rotated");
var loader = document.getElementById("loader");
// correlated model results
var predprobability_randomforest = document.getElementById("randomforest_probability");
var result_randomforest = document.getElementById("randomforest_prediction");
var predprobability__KNN = document.getElementById("KNN_probability");
var result_KNN = document.getElementById("KNN_prediction");


//========================================================================
// Main button events
//========================================================================

function submitImage() {
  // action for the submit button
  console.log("submit");

  if (!imageDisplay.src || !imageDisplay.src.startsWith("data")) {
    window.alert("Please select an image before submit.");
    return;
  }

  loader.classList.remove("hidden");
  imageDisplay.classList.add("loading");


  // call the predict function of the backend
  predictImage(imageDisplay.src);
}

function clearImage() {
  // reset selected files
  fileSelect.value = "";

  // remove image sources and hide them
  imagePreview.src = "";
  imageDisplay.src = "";
  predResult.innerHTML = "";
  predprobability.innerHTML = "";
  imageDisplay_segmented_NoRotated.src ="";
  predResult_segmeted_NoRotated.innerHTML ="";
  predprobability_segmeted_NoRotated.innerHTML ="";
  imageDisplay_rotated.src ="";
  predResult_rotated.innerHTML ="";
  predprobability_rotated.innerHTML ="";



  hide(imagePreview);
  hide(imageDisplay);
  hide(loader);
  hide(predResult);
  hide(predprobability);
  show(uploadCaption);

  hide(imageDisplay_segmented_NoRotated);
  hide(predResult_segmeted_NoRotated);
  hide(predprobability_segmeted_NoRotated);

  hide(imageDisplay_rotated);
  hide(predResult_rotated);
  hide(predprobability_rotated);


  imageDisplay.classList.remove("loading");
  imageDisplay_segmented_NoRotated.classList.remove("loading");
  imageDisplay_rotated.classList.remove("loading")

}

function previewFile(file) {
  // show the preview of the image
  console.log(file.name);
  var fileName = encodeURI(file.name);

  var reader = new FileReader();
  reader.readAsDataURL(file);
  reader.onloadend = () => {
    imagePreview.src = URL.createObjectURL(file);

    show(imagePreview);
    hide(uploadCaption);

    // reset
    predResult.innerHTML = "";
    predResult_segmeted_NoRotated.innerHTML ="";
    predResult_rotated.innerHTML="";

    imageDisplay.classList.remove("loading");
    imageDisplay_segmented_NoRotated.classList.remove("loading")
    imageDisplay_rotated.classList.remove("loading")

    displayImage(reader.result, "image-display");
//    displayImage(reader.result, "image-display_segmented");

  };
}

//========================================================================
// Helper functions
//========================================================================

function predictImage(image) {
  fetch("/predict", {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify(image)
  })
    .then(resp => {
      if (resp.ok)
        resp.json().then(data => {
          displayResult(data);
          imageDisplay_segmented_NoRotated.src = 'static/SaveMask/segmented_mask.jpeg'
          imageDisplay_rotated.src = 'static/SaveSegmentedLungOPenCV/Segmented_lung_openCV.jpeg'
//'static/SaveSegmentedLungOPenCV/Segmented_lung_openCV.jpeg'
        });
    })
    .catch(err => {
      console.log("An error occured", err.message);
      window.alert("Oops! Something went wrong.");
    });
}

function displayImage(image, id) {
  // display image on given id <img> element
  let display = document.getElementById(id);
  display.src = image;
  show(display);
}

function displayResult(data) {
  // display the result
  // imageDisplay.classList.remove("loading");
  console.log(data)
  hide(loader);
  predResult.innerHTML = data.result;
  show(predResult);
  predprobability.innerHTML = data.probability;
  show(predprobability);
  predResult_segmeted_NoRotated.innerHTML = data.result_no_rotated;
  show(predResult_segmeted_NoRotated);
  predprobability_segmeted_NoRotated.innerHTML = data.probability_no_rotated;
  show(predprobability_segmeted_NoRotated)
  predResult_rotated.innerHTML = data.result_segmented_manually;
  show(predResult_rotated);
  predprobability_rotated.innerHTML = data.pred_proba_segmented_manually;
  show(predprobability_rotated)

  // rainforest
predprobability_randomforest.innerHTML = data.prediction_prob_randomforest;
show(predprobability_randomforest)
result_randomforest.innerHTML = data.pred_result_randomForest
show(result_randomforest)

  // KNN
predprobability__KNN.innerHTML = data.prediction_prob_KNN;
show(predprobability__KNN)
result_KNN.innerHTML = data.pred_result_KNN;
show(result_KNN)


}

function hide(el) {
  // hide an element
  el.classList.add("hidden");
}

function show(el) {
  // show an element
  el.classList.remove("hidden");
}