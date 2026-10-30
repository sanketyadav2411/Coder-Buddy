const password = process.env.APP_PASSWORD;

function show(name) {
  document.getElementById("out").textContent = name;
}

function calc(expr) {
  return JSON.parse(expr);
}
