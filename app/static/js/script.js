const button = document.querySelector("#primary");
let previewImage = document.querySelector(".preview-image > img");
let original = document.querySelector(".original-image");
let originalImage = document.querySelector(".original-image > img");
const imageInput = document.querySelector(".image-upload");
let preview = document.querySelector(".preview-image");
const save = document.querySelector("#save");

let open = document.querySelector("#open");
const effectName = document.getElementsByName("effect");
const effectButtons = document.querySelectorAll(".effect-label");
const effectInputs = document.querySelectorAll(".effect-input");
const loader = document.querySelector(".loader");
const navbar = document.querySelector(".bg-chocolate > .nav-tabs");
const tabPanes = navbar.querySelectorAll(".nav-link");
const imageColumn = navbar.querySelector("#col-image");
const textCharacter = document.querySelector(".text-character");

let originImage = null;
let originImagebase64 = null;

sideNavHandler();
toggleAllEditAccordion("hide");
previewImage.addEventListener("load", function (e) {
  toggleAllEditAccordion("show");
  if (!originImage) {
    originImagebase64 = previewImage.src;
    originImage = originImagebase64.split("base64,")[1];
  }
  toggleAllEffectAccordion();
  save.removeAttribute("disabled");
});

function toggleAllEditAccordion(toggleType) {
  const accordionEdit = document.querySelector("#edit");
  const cardHeaders = accordionEdit.querySelectorAll(".card");
  if (toggleType === "hide") {
    cardHeaders.forEach(
      (item) => (
        (item.querySelector(".card-header").style.cursor = "not-allowed"),
        (item.querySelector(".card-header").style.backgroundColor =
          "lightgrey"),
        item.querySelector(".card-header").removeAttribute("data-target")
      )
    );
  } else {
    cardHeaders.forEach((item) => {
      const cardHeader = item.querySelector(".card-header");
      const dataTarget = cardHeader.getAttribute("aria-controls");
      return (
        (cardHeader.style.cursor = "pointer"),
        (cardHeader.style.backgroundColor = "white"),
        cardHeader.setAttribute("data-target", `#${dataTarget}`)
      );
    });
  }
}

function toggleAllEffectAccordion() {
  effectButtons.forEach(
    (item) => (
      item.removeAttribute("disabled"),
      item.classList.remove("disabled"),
      item.addEventListener("click", getRadioButton)
    )
  );
  effectInputs.forEach(
    (item) => (
      item.removeAttribute("disabled"), item.classList.remove("disabled")
    )
  );
}
let imageUploadCount = 0; // Variable to track the number of image uploads
open.addEventListener("click", function () {
  if (imageUploadCount === 1) {
    response = prompt("Did you save the image? Type 'yes' or 'no'");
    if (response) {
      response = response.trim().toLowerCase(); // Trim and convert to lowercase
      if (response === "yes") {
        window.location.reload(true); // true for hard reload
      } else {
        alert("Please save the image first.");
        return;
      }
    } else {
      alert("Please provide a response.");
      return;
    }
  }
  imageInput.click();
  imageUploadCount++;
});

imageInput.addEventListener("change", function (input) {
  if (input.target.files && input.target.files[0]) {
    const fileReader = new FileReader();
    fileReader.onload = function (e) {
      previewImage.setAttribute("src", e.target.result);
      originalImage.setAttribute("src", e.target.result);
      original.style.display = "flex";
      preview.style.display = "flex";
      imageInput.style.display = "none";

      // Increase the image upload count

      // Reload the page if it's the second image upload
    };
    fileReader.readAsDataURL(input.target.files[0]);
  }
});

function sideNavHandler() {
  tabPanes.forEach((item) => {
    item.addEventListener("click", function () {
      if (!item.id === "text-tab") {
        textCharacter.style.display = "none";
      }
    });
  });
}

save.addEventListener("click", function () {
  //   html2canvas(document.querySelector(".col-image")).then(function (canvas) {
  //     saveAs(canvas.toDataURL(), "file-name.png");
  //   });
  // save the image to the file system
  downloadImage = function (data, filename = "untitled.png") {
    let a = document.createElement("a");
    a.href = data;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
  };
  downloadImage(previewImage.src, "image.png");
});

function saveAs(uri, filename) {
  var link = document.createElement("a");
  if (typeof link.download === "string") {
    link.href = uri;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  } else {
    window.open(uri);
  }
}
