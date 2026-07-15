async function uploadPDF(){

let file=document.getElementById("pdf").files[0];

let formData=new FormData();

formData.append("pdf",file);

await fetch("/upload",{
method:"POST",
body:formData
});

alert("PDF Indexed Successfully");

loadStats();
}

function fillQuestion(text){

document.getElementById("question").value=text;
}

async function askQuestion(){

let question=document.getElementById("question").value;

let response=await fetch("/ask",{

method:"POST",

headers:{
"Content-Type":"application/json"
},

body:JSON.stringify({
question:question
})

});

let data=await response.json();

document.getElementById("answerBox").innerHTML=
"<h3>Answer</h3><p>"+data.answer+"</p>";

document.getElementById("contextBox").innerHTML=
"<h3>Retrieved Context</h3><p>"+data.context+"</p>";

loadStats();
}

async function loadStats(){

let response=await fetch("/stats");

let data=await response.json();

document.getElementById("queries").innerText=data.queries;
document.getElementById("documents").innerText=data.documents;
document.getElementById("chunks").innerText=data.chunks;
}

loadStats();