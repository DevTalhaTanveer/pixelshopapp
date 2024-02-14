const cropAccordion = document.querySelector("#crop-accordion");
const rotateAccordion = document.querySelector("#rotate-accordion");
const collapseOne = document.querySelector("#collapseOne");
const applyCrop = document.querySelector("#apply-crop");
const applyRotate = document.querySelector("#apply-rotate");
const applyResize = document.querySelector("#apply-resize");
const reset = document.querySelector("#reset");
const editCloseButtons = document.querySelectorAll("#edit-close-button");
const flipHorizontal = document.querySelector("#flip-horizontal");
const flipVertical = document.querySelector("#flip-vertical");
const rotateRight = document.querySelector("#rotate-right");
const rotateLeft = document.querySelector("#rotate-left");
const brightnessInput = document.querySelector("#brightness-input");
let brightnessLabel = document
  .querySelector("#brightness-label")
  .querySelector(".value");
const contrastInput = document.querySelector("#contrast-input");
let contrastLabel = document
  .querySelector("#contrast-label")
  .querySelector(".value");
const sharpnessInput = document.querySelector("#sharpness-input");
let sharpnessLabel = document
  .querySelector("#sharpness-label")
  .querySelector(".value");
const saturationInput = document.querySelector("#saturation-input");
const saturationLabel = document
  .querySelector("#saturation-label")
  .querySelector(".value");
const exposureInput = document.querySelector("#exposure-input");
const exposureLabel = document
  .querySelector("#exposure-label")
  .querySelector(".value");
const filterInput = document.querySelector("#filter-input");
const filterLabel = document
  .querySelector("#filter-label")
  .querySelector(".value");
const greenInput = document.querySelector("#green-button");
const blueInput = document.querySelector("#blue-button");
const redInput = document.querySelector("#red-button");
let greenChanged = false;
let blueChanged = false;
let redChanged = false;
const greenLabel = document
  .querySelector("#green-label")
  .querySelector(".value");
const greenhue = document.querySelector("#green-hue");
const greenhueLabel = document
  .querySelector("#green-hue-label")
  .querySelector(".value");
const green_saturation_1 = document.querySelector("#green-saturation-1");
const green_saturation_2 = document.querySelector("#green-saturation-2");
let green_saturation_1_label = document
  .querySelector("#green-saturation-1-label")
  .querySelector(".value");
let green_saturation_2_label = document
  .querySelector("#green-saturation-2-label")
  .querySelector(".value");
const green_saturation_1_input_number=document.querySelector("#green-input-1")
const green_saturation_2_input_number=document.querySelector("#green-input-2")
const blue_saturation_1_input_number=document.querySelector("#blue-input-1")
const blue_saturation_2_input_number=document.querySelector("#blue-input-2")
const red_saturation_1_input_number=document.querySelector("#red-input-1")
const red_saturation_2_input_number=document.querySelector("#red-input-2")

// Now we add the logic of input feilds 


const bluehue = document.querySelector("#blue-hue");
const bluehueLabel = document
  .querySelector("#blue-hue-label")
  .querySelector(".value");
const blue_saturation_1 = document.querySelector("#blue-saturation-1");
const blue_saturation_2 = document.querySelector("#blue-saturation-2");
const blue_saturation_1_label = document
  .querySelector("#blue-saturation-1-label")
  .querySelector(".value");
const blue_saturation_2_label = document
  .querySelector("#blue-saturation-2-label")
  .querySelector(".value");
const redhue = document.querySelector("#red-hue");
const redhueLabel = document
  .querySelector("#red-hue-label")
  .querySelector(".value");
const red_saturation_1 = document.querySelector("#red-saturation-1");
const red_saturation_2 = document.querySelector("#red-saturation-2");
const red_saturation_1_label = document
  .querySelector("#red-saturation-1-label")
  .querySelector(".value");
const red_saturation_2_label = document
  .querySelector("#red-saturation-2-label")
  .querySelector(".value");

const height = document.querySelector("#height");
const width = document.querySelector("#width");
// const show = collapseOne.classList.contains("show")
let cropper;
const ctx_g = document.getElementById("histogram").getContext("2d");
const ctx_r = document.getElementById("histogram_r").getContext("2d");
const ctx_b = document.getElementById("histogram_b").getContext("2d");
const labels = Array.from({ length: 256 }, (_, index) => index);
console.log(labels);

$(document).click(function (event) {
  var $target = $(event.target);
  if (!$target.closest(".text-box").length && $(".text-box").is(":visible")) {
    $(".resizers").hide();
  }
});

function closeButtonHandler(e) {
  const parent = e.target.parentNode.parentNode.parentNode.parentNode;
  parent.classList.remove("show");
  cropper.destroy();
}

editCloseButtons.forEach((b) =>
  b.addEventListener("click", closeButtonHandler)
);
// now we add the the logic of up scale the image with ai 
let upscalebtn=document.querySelector("#edit > button")
console.log(upscalebtn);
// end decleration
document.querySelector("#headingFour > h5")




/**OTHERS */
brightnessInput.addEventListener("change", function (e) {
  let response = originImage;
  let contrast = Number(contrastLabel.innerHTML);
  let sharpness = Number(sharpnessLabel.innerHTML);
  let saturation = Number(saturationLabel.innerHTML);
  let exposure = Number(exposureLabel.innerHTML);
  let filter = Number(filterLabel.innerHTML);
  let bluehue = Number(bluehueLabel.innerHTML);
  let blue_saturation_1 = Number(blue_saturation_1_label.innerHTML);
  let blue_saturation_2 = Number(blue_saturation_2_label.innerHTML);
  let red_hue = Number(redhueLabel.innerHTML);
  let red_saturation_1 = Number(red_saturation_1_label.innerHTML);
  let red_saturation_2 = Number(red_saturation_2_label.innerHTML);
  let green_hue = Number(greenhueLabel.innerHTML);
  let green_saturation_1 = Number(green_saturation_1_label.innerHTML);
  let green_saturation_2 = Number(green_saturation_2_label.innerHTML);

  if (
    contrast > 1 ||
    sharpness > 1 ||
    saturation > 1 ||
    exposure > 1 ||
    filter < 1 ||
    bluehue > 0 ||
    blue_saturation_1 > 0 ||
    blue_saturation_2 < 255 ||
    red_hue > 0 ||
    red_saturation_1 > 0 ||
    red_saturation_2 < 255 ||
    green_hue > 0 ||
    green_saturation_1 > 0 ||
    green_saturation_2 < 255 ||
    contrast < 1 ||
    sharpness < 1 ||
    saturation < 1 ||
    exposure < 1
  ) {
    console.log("a");
    response = previewImage.src;
    response = response.split("base64,")[1];
  }

  loader.style.display = "block";
  const value = e.target.value;
  console.log(value);
  brightnessLabel.innerHTML = value;
  let factorial = Number(value);
  axios
    .post(
      "/others/brightness",
      {
        response,
        factorial,
      },
      {
        headers: {
          "Content-Type": "application/json",
        },
      }
    )
    .then((res) => {
      const image = res.data.res;
      let result = image.split("'")[1];
      previewImage.setAttribute("src", "data:image/png;base64," + result);
      loader.style.display = "none";
    })
    .catch((err) => console.error(err));
});

contrastInput.addEventListener("change", function (e) {
  let response = originImage;
  let brightness = Number(brightnessLabel.innerHTML);
  let sharpness = Number(sharpnessLabel.innerHTML);
  let saturation = Number(saturationLabel.innerHTML);
  let exposure = Number(exposureLabel.innerHTML);
  let filter = Number(filterLabel.innerHTML);
  let bluehue = Number(bluehueLabel.innerHTML);
  let blue_saturation_1 = Number(blue_saturation_1_label.innerHTML);
  let blue_saturation_2 = Number(blue_saturation_2_label.innerHTML);
  let red_hue = Number(redhueLabel.innerHTML);
  let red_saturation_1 = Number(red_saturation_1_label.innerHTML);
  let red_saturation_2 = Number(red_saturation_2_label.innerHTML);
  let green_hue = Number(greenhueLabel.innerHTML);
  let green_saturation_1 = Number(green_saturation_1_label.innerHTML);
  let green_saturation_2 = Number(green_saturation_2_label.innerHTML);

  if (
    brightness > 1 ||
    sharpness > 1 ||
    saturation > 1 ||
    exposure > 1 ||
    filter < 1 ||
    bluehue > 0 ||
    blue_saturation_1 > 0 ||
    blue_saturation_2 < 255 ||
    red_hue > 0 ||
    red_saturation_1 > 0 ||
    red_saturation_2 < 255 ||
    green_hue > 0 ||
    green_saturation_1 > 0 ||
    green_saturation_2 < 255 ||
    brightness < 1 ||
    sharpness < 1 ||
    saturation < 1 ||
    exposure < 1
  ) {
    response = previewImage.src;
    response = response.split("base64,")[1];
  }
  loader.style.display = "block";
  const value = e.target.value;
  contrastLabel.innerHTML = value;
  let factorial = Number(value);

  axios
    .post(
      "/others/contrast",
      {
        response,
        factorial,
      },
      {
        headers: {
          "Content-Type": "application/json",
        },
      }
    )
    .then((res) => {
      const image = res.data.res;
      let result = image.split("'")[1];
      previewImage.setAttribute("src", "data:image/png;base64," + result);
      loader.style.display = "none";
    })
    .catch((err) => console.error(err));
});

sharpnessInput.addEventListener("change", function (e) {
  loader.style.display = "block";
  const value = e.target.value;
  sharpnessLabel.innerHTML = value;
  console.log(value);
  let factorial = Number(value);
  let response = originImage;
  let contrast = Number(contrastLabel.innerHTML);
  let brightness = Number(brightnessLabel.innerHTML);
  let saturation = Number(saturationLabel.innerHTML);
  let exposure = Number(exposureLabel.innerHTML);
  let filter = Number(filterLabel.innerHTML);
  let bluehue = Number(bluehueLabel.innerHTML);
  let blue_saturation_1 = Number(blue_saturation_1_label.innerHTML);
  let blue_saturation_2 = Number(blue_saturation_2_label.innerHTML);
  let red_hue = Number(redhueLabel.innerHTML);
  let red_saturation_1 = Number(red_saturation_1_label.innerHTML);
  let red_saturation_2 = Number(red_saturation_2_label.innerHTML);
  let green_hue = Number(greenhueLabel.innerHTML);
  let green_saturation_1 = Number(green_saturation_1_label.innerHTML);
  let green_saturation_2 = Number(green_saturation_2_label.innerHTML);

  if (
    contrast > 1 ||
    brightness > 1 ||
    saturation > 1 ||
    exposure > 1 ||
    filter < 1 ||
    bluehue > 0 ||
    blue_saturation_1 > 0 ||
    blue_saturation_2 < 255 ||
    red_hue > 0 ||
    red_saturation_1 > 0 ||
    red_saturation_2 < 255 ||
    green_hue > 0 ||
    green_saturation_1 > 0 ||
    green_saturation_2 < 255 ||
    contrast < 1 ||
    brightness < 1 ||
    saturation < 1 ||
    exposure < 1
  ) {
    response = previewImage.src;
    response = response.split("base64,")[1];
  }

  axios
    .post(
      "/others/sharpness",
      {
        response,
        factorial,
      },
      {
        headers: {
          "Content-Type": "application/json",
        },
      }
    )
    .then((res) => {
      const image = res.data.res;
      let result = image.split("'")[1];
      previewImage.setAttribute("src", "data:image/png;base64," + result);
      loader.style.display = "none";
    })
    .catch((err) => console.error(err));
});

saturationInput.addEventListener("change", function (e) {
  loader.style.display = "block";
  const value = e.target.value;
  saturationLabel.innerHTML = value;
  let response = originImage;
  let contrast = Number(contrastLabel.innerHTML);
  let brightness = Number(brightnessLabel.innerHTML);
  let sharpness = Number(sharpnessLabel.innerHTML);
  let exposure = Number(exposureLabel.innerHTML);
  let filter = Number(filterLabel.innerHTML);
  let bluehue = Number(bluehueLabel.innerHTML);
  let blue_saturation_1 = Number(blue_saturation_1_label.innerHTML);
  let blue_saturation_2 = Number(blue_saturation_2_label.innerHTML);
  let red_hue = Number(redhueLabel.innerHTML);
  let red_saturation_1 = Number(red_saturation_1_label.innerHTML);
  let red_saturation_2 = Number(red_saturation_2_label.innerHTML);
  let green_hue = Number(greenhueLabel.innerHTML);
  let green_saturation_1 = Number(green_saturation_1_label.innerHTML);
  let green_saturation_2 = Number(green_saturation_2_label.innerHTML);

  if (
    contrast > 1 ||
    brightness > 1 ||
    sharpness > 1 ||
    exposure > 1 ||
    filter < 1 ||
    bluehue > 0 ||
    blue_saturation_1 > 0 ||
    blue_saturation_2 < 255 ||
    red_hue > 0 ||
    red_saturation_1 > 0 ||
    red_saturation_2 < 255 ||
    green_hue > 0 ||
    green_saturation_1 > 0 ||
    green_saturation_2 < 255 ||
    contrast < 1 ||
    brightness < 1 ||
    sharpness < 1 ||
    exposure < 1
  ) {
    response = previewImage.src;
    response = response.split("base64,")[1];
  }

  let factorial = Number(value);

  axios
    .post(
      "/color/saturation",
      {
        response,
        factorial,
      },
      {
        headers: {
          "Content-Type": "application/json",
        },
      }
    )
    .then((res) => {
      const image = res.data.res;
      let result = image.split("'")[1];
      previewImage.setAttribute("src", "data:image/png;base64," + result);
      loader.style.display = "none";
    })
    .catch((err) => console.error(err));
});

exposureInput.addEventListener("change", function (e) {
  console.log("exposure");
  loader.style.display = "block";
  const value = e.target.value;
  exposureLabel.innerHTML = value;
  console.log(value);
  let response = originImage;
  let contrast = Number(contrastLabel.innerHTML);
  let brightness = Number(brightnessLabel.innerHTML);
  let sharpness = Number(sharpnessLabel.innerHTML);
  let saturation = Number(saturationLabel.innerHTML);
  let filter = Number(filterLabel.innerHTML);
  let bluehue = Number(bluehueLabel.innerHTML);
  let blue_saturation_1 = Number(blue_saturation_1_label.innerHTML);
  let blue_saturation_2 = Number(blue_saturation_2_label.innerHTML);
  let red_hue = Number(redhueLabel.innerHTML);
  let red_saturation_1 = Number(red_saturation_1_label.innerHTML);
  let red_saturation_2 = Number(red_saturation_2_label.innerHTML);
  let green_hue = Number(greenhueLabel.innerHTML);
  let green_saturation_1 = Number(green_saturation_1_label.innerHTML);
  let green_saturation_2 = Number(green_saturation_2_label.innerHTML);

  if (
    contrast > 1 ||
    brightness > 1 ||
    sharpness > 1 ||
    saturation > 1 ||
    filter < 1 ||
    bluehue > 0 ||
    blue_saturation_1 > 0 ||
    blue_saturation_2 < 255 ||
    red_hue > 0 ||
    red_saturation_1 > 0 ||
    red_saturation_2 < 255 ||
    green_hue > 0 ||
    green_saturation_1 > 0 ||
    green_saturation_2 < 255 ||
    contrast < 1 ||
    brightness < 1 ||
    sharpness < 1 ||
    saturation < 1
  ) {
    response = previewImage.src;
    response = response.split("base64,")[1];
  }

  let factorial = Number(value);

  axios
    .post(
      "/color/exposure",
      {
        response,
        factorial,
      },
      {
        headers: {
          "Content-Type": "application/json",
        },
      }
    )
    .then((res) => {
      const image = res.data.res;
      let result = image.split("'")[1];
      previewImage.setAttribute("src", "data:image/png;base64," + result);
      loader.style.display = "none";
    })
    .catch((err) => console.error(err));
});

filterInput.addEventListener("change", function (e) {
  loader.style.display = "block";
  const value = e.target.value;
  filterLabel.innerHTML = value;
  console.log(value);
  let response = originImage;
  let contrast = Number(contrastLabel.innerHTML);
  let brightness = Number(brightnessLabel.innerHTML);
  let sharpness = Number(sharpnessLabel.innerHTML);
  let saturation = Number(saturationLabel.innerHTML);
  let exposure = Number(exposureLabel.innerHTML);
  let bluehue = Number(bluehueLabel.innerHTML);
  let blue_saturation_1 = Number(blue_saturation_1_label.innerHTML);
  let blue_saturation_2 = Number(blue_saturation_2_label.innerHTML);
  let red_hue = Number(redhueLabel.innerHTML);
  let red_saturation_1 = Number(red_saturation_1_label.innerHTML);
  let red_saturation_2 = Number(red_saturation_2_label.innerHTML);
  let green_hue = Number(greenhueLabel.innerHTML);
  let green_saturation_1 = Number(green_saturation_1_label.innerHTML);
  let green_saturation_2 = Number(green_saturation_2_label.innerHTML);

  if (
    contrast > 1 ||
    brightness > 1 ||
    sharpness > 1 ||
    saturation > 1 ||
    exposure > 1 ||
    bluehue > 0 ||
    blue_saturation_1 > 0 ||
    blue_saturation_2 < 255 ||
    red_hue > 0 ||
    red_saturation_1 > 0 ||
    red_saturation_2 < 255 ||
    green_hue > 0 ||
    green_saturation_1 > 0 ||
    green_saturation_2 < 255 ||
    contrast < 1 ||
    brightness < 1 ||
    sharpness < 1 ||
    saturation < 1 ||
    exposure < 1
  ) {
    response = previewImage.src;
    response = response.split("base64,")[1];
  }

  let factorial = Number(value);

  axios
    .post(
      "/color/filter",
      {
        response,
        factorial,
      },
      {
        headers: {
          "Content-Type": "application/json",
        },
      }
    )
    .then((res) => {
      const image = res.data.res;
      let result = image.split("'")[1];
      previewImage.setAttribute("src", "data:image/png;base64," + result);
      loader.style.display = "none";
    })
    .catch((err) => console.error(err));
});

greenhue.addEventListener("change", function (e) {
  loader.style.display = "block";
  const value = e.target.value;
  greenhueLabel.innerHTML = value;
  console.log(value);
  let response = originImage;
  let contrast = Number(contrastLabel.innerHTML);
  let brightness = Number(brightnessLabel.innerHTML);
  let sharpness = Number(sharpnessLabel.innerHTML);
  let saturation = Number(saturationLabel.innerHTML);
  let exposure = Number(exposureLabel.innerHTML);
  let filter = Number(filterLabel.innerHTML);
  let bluehue = Number(bluehueLabel.innerHTML);
  let blue_saturation_1 = Number(blue_saturation_1_label.innerHTML);
  let blue_saturation_2 = Number(blue_saturation_2_label.innerHTML);
  let red_hue = Number(redhueLabel.innerHTML);
  let red_saturation_1 = Number(red_saturation_1_label.innerHTML);
  let red_saturation_2 = Number(red_saturation_2_label.innerHTML);
  let green_saturation_1 = Number(green_saturation_1_label.innerHTML);
  let green_saturation_2 = Number(green_saturation_2_label.innerHTML);

  if (
    contrast > 1 ||
    brightness > 1 ||
    sharpness > 1 ||
    saturation > 1 ||
    exposure > 1 ||
    filter < 1 ||
    bluehue > 0 ||
    blue_saturation_1 > 0 ||
    blue_saturation_2 < 255 ||
    red_hue > 0 ||
    red_saturation_1 > 0 ||
    red_saturation_2 < 255 ||
    green_saturation_1 > 0 ||
    green_saturation_2 < 255 ||
    contrast < 1 ||
    brightness < 1 ||
    sharpness < 1 ||
    saturation < 1 ||
    exposure < 1
  ) {
    response = previewImage.src;
    response = response.split("base64,")[1];
  }

  let factorial = Number(value);

  axios
    .post(
      "/hue/green",
      {
        response,
        factorial,
      },
      {
        headers: {
          "Content-Type": "application/json",
        },
      }
    )
    .then((res) => {
      console.log(res);
      const image = res.data.res;
      let result = image.split("'")[1];
      previewImage.setAttribute("src", "data:image/png;base64," + result);

      loader.style.display = "none";
    })
    .catch((err) => console.error(err));
});

green_saturation_1_input_number.addEventListener('input',(e)=>{
    const inputValue = parseInt(e.target.value);
    const maxValue = parseInt(e.target.max);
    const minValue = parseInt(e.target.min);

    if (isNaN(inputValue) || inputValue > maxValue || inputValue < minValue) {
        green_saturation_1_input_number.value = isNaN(inputValue) ? '' : (inputValue > maxValue ? maxValue : minValue);
    }
  loader.style.display = "block";
  const value = e.target.value;
  green_saturation_1_label.innerHTML=value
  green_saturation_1.value=value
  console.log(green_saturation_1_label.innerHTML);
  let response = originImage;
  let contrast = Number(contrastLabel.innerHTML);
  let brightness = Number(brightnessLabel.innerHTML);
  let sharpness = Number(sharpnessLabel.innerHTML);
  let saturation = Number(saturationLabel.innerHTML);
  let exposure = Number(exposureLabel.innerHTML);
  let filter = Number(filterLabel.innerHTML);
  let green_hue = Number(greenhueLabel.innerHTML);
  let green_saturation_2 = Number(green_saturation_2_label.innerHTML);
  let red_hue = Number(redhueLabel.innerHTML);
  let red_saturation_1 = Number(red_saturation_1_label.innerHTML);
  let red_saturation_2 = Number(red_saturation_2_label.innerHTML);
  let bluehue = Number(bluehueLabel.innerHTML);
  let blue_saturation_1 = Number(blue_saturation_1_label.innerHTML);
  let blue_saturation_2 = Number(blue_saturation_2_label.innerHTML);

  if (
    contrast > 1 ||
    brightness > 1 ||
    sharpness > 1 ||
    saturation > 1 ||
    exposure > 1 ||
    filter < 1 ||
    green_hue > 0 ||
    green_saturation_2 < 255 ||
    red_hue > 0 ||
    red_saturation_1 > 0 ||
    red_saturation_2 < 255 ||
    bluehue > 0 ||
    blue_saturation_1 > 0 ||
    blue_saturation_2 < 255 ||
    contrast < 1 ||
    brightness < 1 ||
    sharpness < 1 ||
    saturation < 1 ||
    exposure < 1
  ) {
    response = previewImage.src;
    response = response.split("base64,")[1];
  }

  let factorial = Number(value);

  axios
    .post(
      "/slider1/green",
      {
        response,
        factorial,
      },
      {
        headers: {
          "Content-Type": "application/json",
        },
      }
    )
    .then((res) => {
      const image = res.data.res;
      let result = image.split("'")[1];
      previewImage.setAttribute("src", "data:image/png;base64," + result);

      loader.style.display = "none";
    })
    .catch((err) => console.error(err));
})

green_saturation_1.addEventListener("change", function (e) {
  loader.style.display = "block";
  const value = e.target.value;
  green_saturation_1_label.innerHTML = value;
  console.log(value);
  green_saturation_1_input_number.value=value
  let response = originImage;
  let contrast = Number(contrastLabel.innerHTML);
  let brightness = Number(brightnessLabel.innerHTML);
  let sharpness = Number(sharpnessLabel.innerHTML);
  let saturation = Number(saturationLabel.innerHTML);
  let exposure = Number(exposureLabel.innerHTML);
  let filter = Number(filterLabel.innerHTML);
  let green_hue = Number(greenhueLabel.innerHTML);
  let green_saturation_2 = Number(green_saturation_2_label.innerHTML);
  let red_hue = Number(redhueLabel.innerHTML);
  let red_saturation_1 = Number(red_saturation_1_label.innerHTML);
  let red_saturation_2 = Number(red_saturation_2_label.innerHTML);
  let bluehue = Number(bluehueLabel.innerHTML);
  let blue_saturation_1 = Number(blue_saturation_1_label.innerHTML);
  let blue_saturation_2 = Number(blue_saturation_2_label.innerHTML);

  if (
    contrast > 1 ||
    brightness > 1 ||
    sharpness > 1 ||
    saturation > 1 ||
    exposure > 1 ||
    filter < 1 ||
    green_hue > 0 ||
    green_saturation_2 < 255 ||
    red_hue > 0 ||
    red_saturation_1 > 0 ||
    red_saturation_2 < 255 ||
    bluehue > 0 ||
    blue_saturation_1 > 0 ||
    blue_saturation_2 < 255 ||
    contrast < 1 ||
    brightness < 1 ||
    sharpness < 1 ||
    saturation < 1 ||
    exposure < 1
  ) {
    response = previewImage.src;
    response = response.split("base64,")[1];
  }

  let factorial = Number(value);

  axios
    .post(
      "/slider1/green",
      {
        response,
        factorial,
      },
      {
        headers: {
          "Content-Type": "application/json",
        },
      }
    )
    .then((res) => {
      const image = res.data.res;
      let result = image.split("'")[1];
      previewImage.setAttribute("src", "data:image/png;base64," + result);

      loader.style.display = "none";
    })
    .catch((err) => console.error(err));
});
green_saturation_2_input_number.addEventListener('input',(e)=>{
  const inputValue = parseInt(e.target.value);
  const maxValue = parseInt(e.target.max);
  const minValue = parseInt(e.target.min);

  if (isNaN(inputValue) || inputValue > maxValue || inputValue < minValue) {
      green_saturation_2_input_number.value = isNaN(inputValue) ? '' : (inputValue > maxValue ? maxValue : minValue);
  }
  loader.style.display = "block";
  const value = e.target.value;
  green_saturation_2_label.innerHTML = value;
  green_saturation_2.value=value
  console.log(value);
  let response = originImage;
  let contrast = Number(contrastLabel.innerHTML);
  let brightness = Number(brightnessLabel.innerHTML);
  let sharpness = Number(sharpnessLabel.innerHTML);
  let saturation = Number(saturationLabel.innerHTML);
  let exposure = Number(exposureLabel.innerHTML);
  let filter = Number(filterLabel.innerHTML);
  let green_hue = Number(greenhueLabel.innerHTML);
  let green_saturation_1 = Number(green_saturation_1_label.innerHTML);
  let red_hue = Number(redhueLabel.innerHTML);
  let red_saturation_1 = Number(red_saturation_1_label.innerHTML);
  let red_saturation_2 = Number(red_saturation_2_label.innerHTML);
  let bluehue = Number(bluehueLabel.innerHTML);
  let blue_saturation_1 = Number(blue_saturation_1_label.innerHTML);
  let blue_saturation_2 = Number(blue_saturation_2_label.innerHTML);

  if (
    contrast > 1 ||
    brightness > 1 ||
    sharpness > 1 ||
    saturation > 1 ||
    exposure > 1 ||
    filter < 1 ||
    green_hue > 0 ||
    green_saturation_1 > 0 ||
    red_hue > 0 ||
    red_saturation_1 > 0 ||
    red_saturation_2 < 255 ||
    bluehue > 0 ||
    blue_saturation_1 > 0 ||
    blue_saturation_2 < 255 ||
    contrast < 1 ||
    brightness < 1 ||
    sharpness < 1 ||
    saturation < 1 ||
    exposure < 1
  ) {
    response = previewImage.src;
    response = response.split("base64,")[1];
  }

  let factorial = Number(value);

  axios
    .post(
      "/slider2/green",
      {
        response,
        factorial,
      },
      {
        headers: {
          "Content-Type": "application/json",
        },
      }
    )
    .then((res) => {
      const image = res.data.res;
      let result = image.split("'")[1];
      previewImage.setAttribute("src", "data:image/png;base64," + result);

      loader.style.display = "none";
    })
    .catch((err) => console.error(err));
})
green_saturation_2.addEventListener("change", function (e) {
  loader.style.display = "block";
  const value = e.target.value;
  green_saturation_2_label.innerHTML = value;
  console.log(value);
  green_saturation_2_input_number.value=value
  let response = originImage;
  let contrast = Number(contrastLabel.innerHTML);
  let brightness = Number(brightnessLabel.innerHTML);
  let sharpness = Number(sharpnessLabel.innerHTML);
  let saturation = Number(saturationLabel.innerHTML);
  let exposure = Number(exposureLabel.innerHTML);
  let filter = Number(filterLabel.innerHTML);
  let green_hue = Number(greenhueLabel.innerHTML);
  let green_saturation_1 = Number(green_saturation_1_label.innerHTML);
  let red_hue = Number(redhueLabel.innerHTML);
  let red_saturation_1 = Number(red_saturation_1_label.innerHTML);
  let red_saturation_2 = Number(red_saturation_2_label.innerHTML);
  let bluehue = Number(bluehueLabel.innerHTML);
  let blue_saturation_1 = Number(blue_saturation_1_label.innerHTML);
  let blue_saturation_2 = Number(blue_saturation_2_label.innerHTML);

  if (
    contrast > 1 ||
    brightness > 1 ||
    sharpness > 1 ||
    saturation > 1 ||
    exposure > 1 ||
    filter < 1 ||
    green_hue > 0 ||
    green_saturation_1 > 0 ||
    red_hue > 0 ||
    red_saturation_1 > 0 ||
    red_saturation_2 < 255 ||
    bluehue > 0 ||
    blue_saturation_1 > 0 ||
    blue_saturation_2 < 255 ||
    contrast < 1 ||
    brightness < 1 ||
    sharpness < 1 ||
    saturation < 1 ||
    exposure < 1
  ) {
    response = previewImage.src;
    response = response.split("base64,")[1];
  }

  let factorial = Number(value);

  axios
    .post(
      "/slider2/green",
      {
        response,
        factorial,
      },
      {
        headers: {
          "Content-Type": "application/json",
        },
      }
    )
    .then((res) => {
      const image = res.data.res;
      let result = image.split("'")[1];
      previewImage.setAttribute("src", "data:image/png;base64," + result);

      loader.style.display = "none";
    })
    .catch((err) => console.error(err));
});

bluehue.addEventListener("change", function (e) {
  console.log("bluehue");
  loader.style.display = "block";
  const value = e.target.value;
  bluehueLabel.innerHTML = value;
  console.log(value);
  let response = originImage;
  let contrast = Number(contrastLabel.innerHTML);
  let brightness = Number(brightnessLabel.innerHTML);
  let sharpness = Number(sharpnessLabel.innerHTML);
  let exposure = Number(exposureLabel.innerHTML);
  let filter = Number(filterLabel.innerHTML);
  let green_hue = Number(greenhueLabel.innerHTML);
  let green_saturation_1 = Number(green_saturation_1_label.innerHTML);
  let green_saturation_2 = Number(green_saturation_2_label.innerHTML);
  let red_hue = Number(redhueLabel.innerHTML);
  let red_saturation_1 = Number(red_saturation_1_label.innerHTML);
  let red_saturation_2 = Number(red_saturation_2_label.innerHTML);
  let blue_saturation_1 = Number(blue_saturation_1_label.innerHTML);
  let blue_saturation_2 = Number(blue_saturation_2_label.innerHTML);

  if (
    contrast > 1 ||
    brightness > 1 ||
    sharpness > 1 ||
    exposure > 1 ||
    filter < 1 ||
    green_hue > 0 ||
    green_saturation_1 > 0 ||
    green_saturation_2 < 255 ||
    red_hue > 0 ||
    red_saturation_1 > 0 ||
    red_saturation_2 < 255 ||
    blue_saturation_1 > 0 ||
    blue_saturation_2 < 255 ||
    contrast < 1 ||
    brightness < 1 ||
    sharpness < 1 ||
    exposure < 1
  ) {
    response = previewImage.src;
    response = response.split("base64,")[1];
  }

  let factorial = Number(value);

  axios
    .post(
      "/hue/blue",
      {
        response,
        factorial,
      },
      {
        headers: {
          "Content-Type": "application/json",
        },
      }
    )
    .then((res) => {
      const image = res.data.res;
      let result = image.split("'")[1];
      previewImage.setAttribute("src", "data:image/png;base64," + result);

      loader.style.display = "none";
    })
    .catch((err) => console.error(err));
});

blue_saturation_1_input_number.addEventListener('input',(e)=>{
  const inputValue = parseInt(e.target.value);
  const maxValue = parseInt(e.target.max);
  const minValue = parseInt(e.target.min);

  if (isNaN(inputValue) || inputValue > maxValue || inputValue < minValue) {
      blue_saturation_1_input_number.value = isNaN(inputValue) ? '' : (inputValue > maxValue ? maxValue : minValue);
  }
  loader.style.display = "block";
  const value = e.target.value;
  blue_saturation_1_label.innerHTML = value;
  blue_saturation_1.value=value
  let response = originImage;
  let contrast = Number(contrastLabel.innerHTML);
  let brightness = Number(brightnessLabel.innerHTML);
  let sharpness = Number(sharpnessLabel.innerHTML);
  let exposure = Number(exposureLabel.innerHTML);
  let filter = Number(filterLabel.innerHTML);
  let blue_hue = Number(bluehueLabel.innerHTML);
  let blue_saturation_2 = Number(blue_saturation_2_label.innerHTML);
  let green_hue = Number(greenhueLabel.innerHTML);
  let green_saturation_1 = Number(green_saturation_1_label.innerHTML);
  let green_saturation_2 = Number(green_saturation_2_label.innerHTML);
  let red_hue = Number(redhueLabel.innerHTML);
  let red_saturation_1 = Number(red_saturation_1_label.innerHTML);
  let red_saturation_2 = Number(red_saturation_2_label.innerHTML);

  if (
    contrast > 1 ||
    brightness > 1 ||
    sharpness > 1 ||
    exposure > 1 ||
    filter < 1 ||
    blue_hue > 0 ||
    blue_saturation_2 < 255 ||
    green_hue > 0 ||
    green_saturation_1 > 0 ||
    green_saturation_2 < 255 ||
    red_hue > 0 ||
    red_saturation_1 > 0 ||
    red_saturation_2 < 255 ||
    contrast < 1 ||
    brightness < 1 ||
    sharpness < 1 ||
    exposure < 1
  ) {
    response = previewImage.src;
    response = response.split("base64,")[1];
  }

  let factorial = Number(value);

  axios
    .post(
      "/slider1/blue",
      {
        response,
        factorial,
      },
      {
        headers: {
          "Content-Type": "application/json",
        },
      }
    )
    .then((res) => {
      const image = res.data.res;
      let result = image.split("'")[1];
      previewImage.setAttribute("src", "data:image/png;base64," + result);

      loader.style.display = "none";
    })
    .catch((err) => console.error(err));

})

blue_saturation_1.addEventListener("change", function (e) {
  loader.style.display = "block";
  const value = e.target.value;
  blue_saturation_1_label.innerHTML = value;
  console.log(value);
  blue_saturation_1_input_number.value=value
  let response = originImage;
  let contrast = Number(contrastLabel.innerHTML);
  let brightness = Number(brightnessLabel.innerHTML);
  let sharpness = Number(sharpnessLabel.innerHTML);
  let exposure = Number(exposureLabel.innerHTML);
  let filter = Number(filterLabel.innerHTML);
  let blue_hue = Number(bluehueLabel.innerHTML);
  let blue_saturation_2 = Number(blue_saturation_2_label.innerHTML);
  let green_hue = Number(greenhueLabel.innerHTML);
  let green_saturation_1 = Number(green_saturation_1_label.innerHTML);
  let green_saturation_2 = Number(green_saturation_2_label.innerHTML);
  let red_hue = Number(redhueLabel.innerHTML);
  let red_saturation_1 = Number(red_saturation_1_label.innerHTML);
  let red_saturation_2 = Number(red_saturation_2_label.innerHTML);

  if (
    contrast > 1 ||
    brightness > 1 ||
    sharpness > 1 ||
    exposure > 1 ||
    filter < 1 ||
    blue_hue > 0 ||
    blue_saturation_2 < 255 ||
    green_hue > 0 ||
    green_saturation_1 > 0 ||
    green_saturation_2 < 255 ||
    red_hue > 0 ||
    red_saturation_1 > 0 ||
    red_saturation_2 < 255 ||
    contrast < 1 ||
    brightness < 1 ||
    sharpness < 1 ||
    exposure < 1
  ) {
    response = previewImage.src;
    response = response.split("base64,")[1];
  }

  let factorial = Number(value);

  axios
    .post(
      "/slider1/blue",
      {
        response,
        factorial,
      },
      {
        headers: {
          "Content-Type": "application/json",
        },
      }
    )
    .then((res) => {
      const image = res.data.res;
      let result = image.split("'")[1];
      previewImage.setAttribute("src", "data:image/png;base64," + result);

      loader.style.display = "none";
    })
    .catch((err) => console.error(err));
});

blue_saturation_2_input_number.addEventListener('input',(e)=>{
  const inputValue = parseInt(e.target.value);
  const maxValue = parseInt(e.target.max);
  const minValue = parseInt(e.target.min);

  if (isNaN(inputValue) || inputValue > maxValue || inputValue < minValue) {
      blue_saturation_2_input_number.value = isNaN(inputValue) ? '' : (inputValue > maxValue ? maxValue : minValue);
  }
  loader.style.display = "block";
  const value = e.target.value;
  blue_saturation_2_label.innerHTML = value;
  blue_saturation_2.value=value
  let response = originImage;
  let contrast = Number(contrastLabel.innerHTML);
  let brightness = Number(brightnessLabel.innerHTML);
  let sharpness = Number(sharpnessLabel.innerHTML);
  let exposure = Number(exposureLabel.innerHTML);
  let filter = Number(filterLabel.innerHTML);
  let blue_hue = Number(bluehueLabel.innerHTML);
  let blue_saturation_1 = Number(blue_saturation_1_label.innerHTML);
  let green_hue = Number(greenhueLabel.innerHTML);
  let green_saturation_1 = Number(green_saturation_1_label.innerHTML);
  let green_saturation_2 = Number(green_saturation_2_label.innerHTML);
  let red_hue = Number(redhueLabel.innerHTML);
  let red_saturation_1 = Number(red_saturation_1_label.innerHTML);
  let red_saturation_2 = Number(red_saturation_2_label.innerHTML);

  if (
    contrast > 1 ||
    brightness > 1 ||
    sharpness > 1 ||
    exposure > 1 ||
    filter < 1 ||
    blue_hue > 0 ||
    blue_saturation_1 > 0 ||
    green_hue > 0 ||
    green_saturation_1 > 0 ||
    green_saturation_2 < 255 ||
    red_hue > 0 ||
    red_saturation_1 > 0 ||
    red_saturation_2 < 255 ||
    contrast < 1 ||
    brightness < 1 ||
    sharpness < 1 ||
    exposure < 1
  ) {
    response = previewImage.src;
    response = response.split("base64,")[1];
  }

  let factorial = Number(value);

  axios
    .post(
      "/slider2/blue",
      {
        response,
        factorial,
      },
      {
        headers: {
          "Content-Type": "application/json",
        },
      }
    )
    .then((res) => {
      const image = res.data.res;
      let result = image.split("'")[1];
      previewImage.setAttribute("src", "data:image/png;base64," + result);

      loader.style.display = "none";
    })
    .catch((err) => console.error(err));
})
blue_saturation_2.addEventListener("change", function (e) {
  loader.style.display = "block";
  const value = e.target.value;
  blue_saturation_2_label.innerHTML = value;
  console.log(value);
  let response = originImage;
  blue_saturation_2_input_number.value=value
  let contrast = Number(contrastLabel.innerHTML);
  let brightness = Number(brightnessLabel.innerHTML);
  let sharpness = Number(sharpnessLabel.innerHTML);
  let exposure = Number(exposureLabel.innerHTML);
  let filter = Number(filterLabel.innerHTML);
  let blue_hue = Number(bluehueLabel.innerHTML);
  let blue_saturation_1 = Number(blue_saturation_1_label.innerHTML);
  let green_hue = Number(greenhueLabel.innerHTML);
  let green_saturation_1 = Number(green_saturation_1_label.innerHTML);
  let green_saturation_2 = Number(green_saturation_2_label.innerHTML);
  let red_hue = Number(redhueLabel.innerHTML);
  let red_saturation_1 = Number(red_saturation_1_label.innerHTML);
  let red_saturation_2 = Number(red_saturation_2_label.innerHTML);

  if (
    contrast > 1 ||
    brightness > 1 ||
    sharpness > 1 ||
    exposure > 1 ||
    filter < 1 ||
    blue_hue > 0 ||
    blue_saturation_1 > 0 ||
    green_hue > 0 ||
    green_saturation_1 > 0 ||
    green_saturation_2 < 255 ||
    red_hue > 0 ||
    red_saturation_1 > 0 ||
    red_saturation_2 < 255 ||
    contrast < 1 ||
    brightness < 1 ||
    sharpness < 1 ||
    exposure < 1
  ) {
    response = previewImage.src;
    response = response.split("base64,")[1];
  }

  let factorial = Number(value);

  axios
    .post(
      "/slider2/blue",
      {
        response,
        factorial,
      },
      {
        headers: {
          "Content-Type": "application/json",
        },
      }
    )
    .then((res) => {
      const image = res.data.res;
      let result = image.split("'")[1];
      previewImage.setAttribute("src", "data:image/png;base64," + result);

      loader.style.display = "none";
    })
    .catch((err) => console.error(err));
});

redhue.addEventListener("change", function (e) {
  console.log("redhue");
  loader.style.display = "block";
  const value = e.target.value;
  redhueLabel.innerHTML = value;
  console.log(value);
  let response = originImage;
  let contrast = Number(contrastLabel.innerHTML);
  let brightness = Number(brightnessLabel.innerHTML);
  let sharpness = Number(sharpnessLabel.innerHTML);
  let exposure = Number(exposureLabel.innerHTML);
  let filter = Number(filterLabel.innerHTML);
  let green_hue = Number(greenhueLabel.innerHTML);
  let green_saturation_1 = Number(green_saturation_1_label.innerHTML);
  let green_saturation_2 = Number(green_saturation_2_label.innerHTML);
  let bluehue = Number(bluehueLabel.innerHTML);
  let blue_saturation_1 = Number(blue_saturation_1_label.innerHTML);
  let blue_saturation_2 = Number(blue_saturation_2_label.innerHTML);
  let red_saturation_1 = Number(red_saturation_1_label.innerHTML);
  let red_saturation_2 = Number(red_saturation_2_label.innerHTML);

  if (
    contrast > 1 ||
    brightness > 1 ||
    sharpness > 1 ||
    exposure > 1 ||
    filter < 1 ||
    red_saturation_1 > 0 ||
    red_saturation_2 < 255 ||
    bluehue > 0 ||
    blue_saturation_1 > 0 ||
    blue_saturation_2 < 255 ||
    green_hue > 0 ||
    green_saturation_1 > 0 ||
    green_saturation_2 < 255 ||
    contrast < 1 ||
    brightness < 1 ||
    sharpness < 1 ||
    exposure < 1
  ) {
    response = previewImage.src;
    response = response.split("base64,")[1];
  }

  let factorial = Number(value);

  axios
    .post(
      "/hue/red",
      {
        response,
        factorial,
      },
      {
        headers: {
          "Content-Type": "application/json",
        },
      }
    )
    .then((res) => {
      const image = res.data.res;
      let result = image.split("'")[1];
      previewImage.setAttribute("src", "data:image/png;base64," + result);

      loader.style.display = "none";
    })
    .catch((err) => console.error(err));
});

red_saturation_1_input_number.addEventListener('input',(e)=>{
  const inputValue = parseInt(e.target.value);
  const maxValue = parseInt(e.target.max);
  const minValue = parseInt(e.target.min);

  if (isNaN(inputValue) || inputValue > maxValue || inputValue < minValue) {
      red_saturation_1_input_number.value = isNaN(inputValue) ? '' : (inputValue > maxValue ? maxValue : minValue);
  }
  loader.style.display = "block";
  const value = e.target.value;
  red_saturation_1_label.innerHTML = value;
  red_saturation_1.value=value
  console.log(value);
  let response = originImage;
  let contrast = Number(contrastLabel.innerHTML);
  let brightness = Number(brightnessLabel.innerHTML);
  let sharpness = Number(sharpnessLabel.innerHTML);

  let exposure = Number(exposureLabel.innerHTML);
  let filter = Number(filterLabel.innerHTML);
  let red_hue = Number(redhueLabel.innerHTML);
  let red_saturation_2 = Number(red_saturation_2_label.innerHTML);
  let bluehue = Number(bluehueLabel.innerHTML);
  let blue_saturation_1 = Number(blue_saturation_1_label.innerHTML);
  let blue_saturation_2 = Number(blue_saturation_2_label.innerHTML);
  let green_hue = Number(greenhueLabel.innerHTML);
  let green_saturation_1 = Number(green_saturation_1_label.innerHTML);
  let green_saturation_2 = Number(green_saturation_2_label.innerHTML);

  if (
    contrast > 1 ||
    brightness > 1 ||
    sharpness > 1 ||
    exposure > 1 ||
    filter < 1 ||
    greenChanged ||
    blueChanged ||
    red_hue > 0 ||
    red_saturation_2 < 255 ||
    bluehue > 0 ||
    blue_saturation_1 > 0 ||
    blue_saturation_2 < 255 ||
    green_hue > 0 ||
    green_saturation_1 > 0 ||
    green_saturation_2 < 255 ||
    contrast < 1 ||
    brightness < 1 ||
    sharpness < 1 ||
    exposure < 1
  ) {
    response = previewImage.src;
    response = response.split("base64,")[1];
  }

  let factorial = Number(value);

  axios
    .post(
      "/slider1/red",
      {
        response,
        factorial,
      },
      {
        headers: {
          "Content-Type": "application/json",
        },
      }
    )
    .then((res) => {
      const image = res.data.res;
      let result = image.split("'")[1];
      previewImage.setAttribute("src", "data:image/png;base64," + result);

      loader.style.display = "none";
    })
    .catch((err) => console.error(err));

})

red_saturation_1.addEventListener("change", function (e) {
  loader.style.display = "block";
  const value = e.target.value;
  red_saturation_1_label.innerHTML = value;
  console.log(value);
  red_saturation_1_input_number.value=value;
  let response = originImage;
  let contrast = Number(contrastLabel.innerHTML);
  let brightness = Number(brightnessLabel.innerHTML);
  let sharpness = Number(sharpnessLabel.innerHTML);

  let exposure = Number(exposureLabel.innerHTML);
  let filter = Number(filterLabel.innerHTML);
  let red_hue = Number(redhueLabel.innerHTML);
  let red_saturation_2 = Number(red_saturation_2_label.innerHTML);
  let bluehue = Number(bluehueLabel.innerHTML);
  let blue_saturation_1 = Number(blue_saturation_1_label.innerHTML);
  let blue_saturation_2 = Number(blue_saturation_2_label.innerHTML);
  let green_hue = Number(greenhueLabel.innerHTML);
  let green_saturation_1 = Number(green_saturation_1_label.innerHTML);
  let green_saturation_2 = Number(green_saturation_2_label.innerHTML);

  if (
    contrast > 1 ||
    brightness > 1 ||
    sharpness > 1 ||
    exposure > 1 ||
    filter < 1 ||
    greenChanged ||
    blueChanged ||
    red_hue > 0 ||
    red_saturation_2 < 255 ||
    bluehue > 0 ||
    blue_saturation_1 > 0 ||
    blue_saturation_2 < 255 ||
    green_hue > 0 ||
    green_saturation_1 > 0 ||
    green_saturation_2 < 255 ||
    contrast < 1 ||
    brightness < 1 ||
    sharpness < 1 ||
    exposure < 1
  ) {
    response = previewImage.src;
    response = response.split("base64,")[1];
  }

  let factorial = Number(value);

  axios
    .post(
      "/slider1/red",
      {
        response,
        factorial,
      },
      {
        headers: {
          "Content-Type": "application/json",
        },
      }
    )
    .then((res) => {
      const image = res.data.res;
      let result = image.split("'")[1];
      previewImage.setAttribute("src", "data:image/png;base64," + result);

      loader.style.display = "none";
    })
    .catch((err) => console.error(err));
});

red_saturation_2_input_number.addEventListener('input',(e)=>{
  const inputValue = parseInt(e.target.value);
  const maxValue = parseInt(e.target.max);
  const minValue = parseInt(e.target.min);

  if (isNaN(inputValue) || inputValue > maxValue || inputValue < minValue) {
      red_saturation_2_input_number.value = isNaN(inputValue) ? '' : (inputValue > maxValue ? maxValue : minValue);
  }
  loader.style.display = "block";
  const value = e.target.value;
  red_saturation_2_label.innerHTML = value;
  red_saturation_2.value=value
  console.log(value);
  let response = originImage;
  let contrast = Number(contrastLabel.innerHTML);
  let brightness = Number(brightnessLabel.innerHTML);
  let sharpness = Number(sharpnessLabel.innerHTML);

  let exposure = Number(exposureLabel.innerHTML);
  let filter = Number(filterLabel.innerHTML);
  let red_hue = Number(redhueLabel.innerHTML);
  let red_saturation_1 = Number(red_saturation_1_label.innerHTML);
  let bluehue = Number(bluehueLabel.innerHTML);
  let blue_saturation_1 = Number(blue_saturation_1_label.innerHTML);
  let blue_saturation_2 = Number(blue_saturation_2_label.innerHTML);
  let green_hue = Number(greenhueLabel.innerHTML);
  let green_saturation_1 = Number(green_saturation_1_label.innerHTML);
  let green_saturation_2 = Number(green_saturation_2_label.innerHTML);

  if (
    contrast > 1 ||
    brightness > 1 ||
    sharpness > 1 ||
    exposure > 1 ||
    filter < 1 ||
    greenChanged ||
    blueChanged ||
    red_hue > 0 ||
    red_saturation_1 > 0 ||
    blue_saturation_1 > 0 ||
    blue_saturation_2 < 255 ||
    green_hue > 0 ||
    green_saturation_1 > 0 ||
    green_saturation_2 < 255 ||
    bluehue > 0 ||
    contrast < 1 ||
    brightness < 1 ||
    sharpness < 1 ||
    exposure < 1
  ) {
    response = previewImage.src;
    response = response.split("base64,")[1];
  }

  let factorial = Number(value);

  axios
    .post(
      "/slider2/red",
      {
        response,
        factorial,
      },
      {
        headers: {
          "Content-Type": "application/json",
        },
      }
    )
    .then((res) => {
      const image = res.data.res;
      let result = image.split("'")[1];
      previewImage.setAttribute("src", "data:image/png;base64," + result);

      loader.style.display = "none";
    })
    .catch((err) => console.error(err));
})
red_saturation_2.addEventListener("change", function (e) {
  loader.style.display = "block";
  const value = e.target.value;
  red_saturation_2_label.innerHTML = value;
  console.log(value);
  red_saturation_2_input_number.value=value;
  let response = originImage;
  let contrast = Number(contrastLabel.innerHTML);
  let brightness = Number(brightnessLabel.innerHTML);
  let sharpness = Number(sharpnessLabel.innerHTML);

  let exposure = Number(exposureLabel.innerHTML);
  let filter = Number(filterLabel.innerHTML);
  let red_hue = Number(redhueLabel.innerHTML);
  let red_saturation_1 = Number(red_saturation_1_label.innerHTML);
  let bluehue = Number(bluehueLabel.innerHTML);
  let blue_saturation_1 = Number(blue_saturation_1_label.innerHTML);
  let blue_saturation_2 = Number(blue_saturation_2_label.innerHTML);
  let green_hue = Number(greenhueLabel.innerHTML);
  let green_saturation_1 = Number(green_saturation_1_label.innerHTML);
  let green_saturation_2 = Number(green_saturation_2_label.innerHTML);

  if (
    contrast > 1 ||
    brightness > 1 ||
    sharpness > 1 ||
    exposure > 1 ||
    filter < 1 ||
    greenChanged ||
    blueChanged ||
    red_hue > 0 ||
    red_saturation_1 > 0 ||
    blue_saturation_1 > 0 ||
    blue_saturation_2 < 255 ||
    green_hue > 0 ||
    green_saturation_1 > 0 ||
    green_saturation_2 < 255 ||
    bluehue > 0 ||
    contrast < 1 ||
    brightness < 1 ||
    sharpness < 1 ||
    exposure < 1
  ) {
    response = previewImage.src;
    response = response.split("base64,")[1];
  }

  let factorial = Number(value);

  axios
    .post(
      "/slider2/red",
      {
        response,
        factorial,
      },
      {
        headers: {
          "Content-Type": "application/json",
        },
      }
    )
    .then((res) => {
      const image = res.data.res;
      let result = image.split("'")[1];
      previewImage.setAttribute("src", "data:image/png;base64," + result);

      loader.style.display = "none";
    })
    .catch((err) => console.error(err));
});
let chart;

greenInput.addEventListener("click", async function (e) {

  if (chart) {
    chart.destroy();
  }
  let histg = [];
  let Sweat_Range = [];
  loader.style.display = "block";

  let contrast = Number(contrastLabel.innerHTML);
  let brightness = Number(brightnessLabel.innerHTML);
  let sharpness = Number(sharpnessLabel.innerHTML);
  let saturation = Number(saturationLabel.innerHTML);
  let exposure = Number(exposureLabel.innerHTML);
  let response = originImage;
  let red_hue = Number(redhueLabel.innerHTML);
  let red_saturation_1 = Number(red_saturation_1_label.innerHTML);
  let red_saturation_2 = Number(red_saturation_2_label.innerHTML);
  let blue_hue = Number(bluehueLabel.innerHTML);
  let blue_saturation_1 = Number(blue_saturation_1_label.innerHTML);
  let blue_saturation_2 = Number(blue_saturation_2_label.innerHTML);
  let green_hue = Number(greenhueLabel.innerHTML);

  if (!greenChanged) {
    greenChanged = true;
  }
  if (
    contrast > 1 ||
    brightness > 1 ||
    sharpness > 1 ||
    saturation > 1 ||
    exposure > 1 ||
    red_hue > 0 ||
    red_saturation_1 > 0 ||
    red_saturation_2 < 255 ||
    blue_hue > 0 ||
    blue_saturation_1 > 0 ||
    blue_saturation_2 < 255 ||
    contrast < 1 ||
    brightness < 1 ||
    sharpness < 1 ||
    saturation < 1 ||
    exposure < 1 ||
    green_hue > 0
  ) {
    response = previewImage.src;
    response = response.split("base64,")[1];
  }

  await axios
    .post(
      "/color/green",
      {
        response,
      },
      {
        headers: {
          "Content-Type": "application/json",
        },
      }
    )
    .then((res) => {
      const image = res.data.res;
      let result = image.split("'")[1];
      previewImage.setAttribute("src", "data:image/png;base64," + result);
      loader.style.display = "none";
      console.log(res.data);
      histg = res.data.hist_g;

      Sweat_Range = res.data.Sweat_Range;

      green_saturation_1.value = res.data.entry_2a;
      green_saturation_2.value = res.data.entry_2b;
      green_saturation_1_label.innerHTML = res.data.entry_2a;
      green_saturation_2_label.innerHTML = res.data.entry_2b;
      green_saturation_1_input_number.value=res.data.entry_2a
      green_saturation_2_input_number.value=res.data.entry_2b
    })
    .catch((err) => console.error(err));
  const graph = document.getElementById("histogram");
  graph.height = 400;
  chart = new Chart(ctx_g, {
    type: "bar",

    data: {
      // 0 to 255
      labels: labels,
      datasets: [
        {
          label: "Green Channel Histogram",
          data: histg,
          backgroundColor: "green",
        },
      ],
    },
  });
  Sweat_Range.map((range, key) => {
    chart.data.datasets.push({
      type: "line",
      label: `Sweet range`,
      data: [
        { x: range[key], y: 0 },
        { x: range[key], y: 255 },
      ],
      borderColor: "blue",
      borderWidth: 2,

      fill: false,
      xAxisID: "x",
      yAxisID: "y",
    });
  });

  chart.update();
});
let chartblue;
blueInput.addEventListener("click", async function (e) {
  let histb = [];
  let Sweat_Range = [];
  loader.style.display = "block";
  if (chartblue) {
    chartblue.destroy();
  }
  let contrast = Number(contrastLabel.innerHTML);
  let brightness = Number(brightnessLabel.innerHTML);
  let sharpness = Number(sharpnessLabel.innerHTML);
  let saturation = Number(saturationLabel.innerHTML);
  let exposure = Number(exposureLabel.innerHTML);
  let green_hue = Number(greenhueLabel.innerHTML);
  let green_saturation_1 = Number(green_saturation_1_label.innerHTML);
  let green_saturation_2 = Number(green_saturation_2_label.innerHTML);
  let red_hue = Number(redhueLabel.innerHTML);
  let red_saturation_1 = Number(red_saturation_1_label.innerHTML);
  let red_saturation_2 = Number(red_saturation_2_label.innerHTML);
  let blue_hue = Number(bluehueLabel.innerHTML);
  let response = originImage;
  if (!blueChanged) {
    blueChanged = true;
  }

  if (
    contrast > 1 ||
    brightness > 1 ||
    sharpness > 1 ||
    saturation > 1 ||
    exposure > 1 ||
    green_hue > 0 ||
    green_saturation_1 > 0 ||
    green_saturation_2 < 255 ||
    red_hue > 0 ||
    red_saturation_1 > 0 ||
    red_saturation_2 < 255 ||
    blue_hue > 0 ||
    contrast < 1 ||
    brightness < 1 ||
    sharpness < 1 ||
    saturation < 1 ||
    exposure < 1
  ) {
    response = previewImage.src;
    response = response.split("base64,")[1];
  }

  await axios
    .post(
      "/color/blue",
      {
        response,
      },
      {
        headers: {
          "Content-Type": "application/json",
        },
      }
    )
    .then((res) => {
      const image = res.data.res;
      let result = image.split("'")[1];
      previewImage.setAttribute("src", "data:image/png;base64," + result);
      loader.style.display = "none";
      histb = res.data.hist_b;
      Sweat_Range = res.data.Sweat_Range;
      blue_saturation_1.value = res.data.entry_3a;
      blue_saturation_2.value = res.data.entry_3b;
      blue_saturation_1_label.innerHTML = res.data.entry_3a;
      blue_saturation_2_label.innerHTML = res.data.entry_3b;
      blue_saturation_1_input_number.value=res.data.entry_3a
      blue_saturation_2_input_number.value=res.data.entry_3b
    })
    .catch((err) => console.error(err));

  console.log(histb);
  const graph = document.getElementById("histogram_b");
  graph.height = 400;

  chartblue = new Chart(ctx_b, {
    type: "bar",
    data: {
      // 0 to 255
      labels: labels,
      datasets: [
        {
          label: "Blue Channel Histogram",
          data: histb,
          backgroundColor: "blue",
        },
      ],
    },
  });
  Sweat_Range.map((range, key) => {
    chartblue.data.datasets.push({
      type: "line",
      label: "sweet range",
      data: [
        { x: range[key], y: 0 },
        { x: range[key], y: 255 },
      ],
      borderColor: "blue",
      borderWidth: 2,

      fill: false,
      xAxisID: "x",
      yAxisID: "y",
    });
  });

  chartblue.update();
});
let chartred;
redInput.addEventListener("click", async function (e) {
  let histr = [];
  let Sweat_Range = [];
  loader.style.display = "block";
  if(chartred){
    chartred.destroy()
  }
  let contrast = Number(contrastLabel.innerHTML);
  let brightness = Number(brightnessLabel.innerHTML);
  let sharpness = Number(sharpnessLabel.innerHTML);
  let saturation = Number(saturationLabel.innerHTML);
  let exposure = Number(exposureLabel.innerHTML);
  let green_hue = Number(greenhueLabel.innerHTML);
  let green_saturation_1 = Number(green_saturation_1_label.innerHTML);
  let green_saturation_2 = Number(green_saturation_2_label.innerHTML);
  let red_hue = Number(redhueLabel.innerHTML);
  let blue_hue = Number(bluehueLabel.innerHTML);
  let blue_saturation_1 = Number(blue_saturation_1_label.innerHTML);
  let blue_saturation_2 = Number(blue_saturation_2_label.innerHTML);
  let response = originImage;

  if (!redChanged) {
    redChanged = true;
  }

  if (
    contrast > 1 ||
    brightness > 1 ||
    sharpness > 1 ||
    saturation > 1 ||
    exposure > 1 ||
    green_hue > 0 ||
    green_saturation_1 > 0 ||
    green_saturation_2 < 255 ||
    blue_hue > 0 ||
    blue_saturation_1 > 0 ||
    blue_saturation_2 < 255 ||
    red_hue > 0 ||
    contrast < 1 ||
    brightness < 1 ||
    sharpness < 1 ||
    saturation < 1 ||
    exposure < 1
  ) {
    console.log("here");
    response = previewImage.src;
    response = response.split("base64,")[1];
  }

  await axios
    .post(
      "/color/red",
      {
        response,
      },
      {
        headers: {
          "Content-Type": "application/json",
        },
      }
    )
    .then((res) => {
      console.log(res);
      const image = res.data.res;
      let result = image.split("'")[1];
      previewImage.setAttribute("src", "data:image/png;base64," + result);
      loader.style.display = "none";
      histr = res.data.hist_r;
      red_saturation_1.value = res.data.entry_1a;
      red_saturation_2.value = res.data.entry_1b;
      Sweat_Range = res.data.Sweat_Range;
      red_saturation_1_label.innerHTML = res.data.entry_1a;
      red_saturation_2_label.innerHTML = res.data.entry_1b;
      red_saturation_1_input_number.value=res.data.entry_1a;
      red_saturation_2_input_number.value=res.data.entry_1b
      console.log(red_saturation_1);
    })
    .catch((err) => console.error(err));

  console.log(Sweat_Range);
  console.log(histr);
  const graph = document.getElementById("histogram_r");
  graph.height = 400;
   chartred= new Chart(ctx_r, {
    type: "bar",
    data: {
      // 0 to 255
      labels: labels,
      datasets: [
        {
          label: "Red Channel Histogram",
          data: histr,
          backgroundColor: "red",
        },
      ],
    },
  });
  Sweat_Range.map((range, key) => {
    chartred.data.datasets.push({
      type: "line",

      label: "sweet range",
      data: [
        { x: range[key], y: 0 },
        { x: range[key], y: 255 },
      ],
      borderColor: "blue",
      borderWidth: 2,

      fill: true,
      xAxisID: "x",
      yAxisID: "y",
    });
  });

  chartred.update();
});
upscalebtn.addEventListener('click',async function (e){
  loader.style.display = "block";
  response=previewImage.src;
  console.log(response);
 response= response.split("base64,")[1];
 await axios
    .post(
      "/upscale",
      {
        response,
      },
      {
        headers: {
          "Content-Type": "application/json",
        },
      }
    )
    .then((res) => {
      loader.style.display = "none";
      const image = res.data.res;
      let result = image
      previewImage.setAttribute("src", "data:image/png;base64," + result);

      
    })
    .catch((err) => console.error(err));


})
reset.addEventListener("click", function (params) {
  brightnessLabel.innerHTML = 1;
  sharpnessLabel.innerHTML = 1;
  saturationLabel.innerHTML = 1;
  contrastLabel.innerHTML = 1;
  brightnessInput.value = 1;
  saturationInput.value = 1;
  sharpnessInput.value = 1;
  contrastInput.value = 1;
  exposureInput.value = 1;
  exposureLabel.innerHTML = 1;
  previewImage.setAttribute("src", originImagebase64);
});
