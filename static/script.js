document.getElementById("predictForm").addEventListener("submit", function(e) {
    e.preventDefault();

    const attendance = document.getElementById("attendance").value;
    const marks = document.getElementById("marks").value;
    const assignment = document.getElementById("assignment").value;

    fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            attendance: attendance,
            internal_marks: marks,
            assignment: assignment
        })
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById("result").innerText =
            "Prediction: " + data.result;
    });
});
