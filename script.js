document.addEventListener("DOMContentLoaded", function () {
    initializeExpansionPanels();
    initializeChatbot();
});

// Selectors
const chatInput = document.querySelector(".chat-input textarea");
const sendChatBtn = document.querySelector(".chat-input button");
const chatbox = document.querySelector(".chatbox");

// OpenAI API Configuration
const API_URL = "https://api.openai.com/v1/chat/completions";
const API_KEY = "";


// Function to create chat messages
const createChatLi = (message, className) => {
    const chatLi = document.createElement("li");
    chatLi.classList.add("chat", className);
    chatLi.innerHTML = `<p>${message}</p>`;
    return chatLi;
};

// Function to handle OpenAI API response
const generateResponse = (incomingChatLi, userMessage) => {
    const messageElement = incomingChatLi.querySelector("p");

    fetch(API_URL, {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
            "Authorization": `Bearer ${API_KEY}`
        },
        body: JSON.stringify({
            model: "gpt-3.5-turbo",
            messages: [{ role: "user", content: userMessage }]
        })
    })
    .then(res => {
        if (!res.ok) throw new Error("Failed to fetch response from OpenAI");
        return res.json();
    })
    .then(data => {
        messageElement.textContent = data.choices[0].message.content;
    })
    .catch(() => {
        messageElement.classList.add("error");
        messageElement.textContent = "❌ Error: Please try again!";
    })
    .finally(() => {
        chatbox.scrollTo(0, chatbox.scrollHeight);
    });
};

// Function to send and display chat messages
const handleChat = () => {
    let userMessage = chatInput.value.trim();
    if (!userMessage) return;

    // Add outgoing message
    chatbox.appendChild(createChatLi(userMessage, "chat-outgoing"));
    chatbox.scrollTo(0, chatbox.scrollHeight);
    
    // Clear input field
    chatInput.value = "";

    // Add "Thinking..." placeholder for incoming response
    setTimeout(() => {
        const incomingChatLi = createChatLi("Thinking...", "chat-incoming");
        chatbox.appendChild(incomingChatLi);
        chatbox.scrollTo(0, chatbox.scrollHeight);

        generateResponse(incomingChatLi, userMessage);
    }, 600);
};

// Event Listeners
sendChatBtn.addEventListener("click", handleChat);

// Enable "Enter" key for sending messages
chatInput.addEventListener("keypress", function (event) {
    if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        handleChat();
    }
});

// Function to close chatbot
function cancel() {
    let chatbotComplete = document.querySelector(".chatBot");
    if (chatbotComplete.style.display !== "none") {
        chatbotComplete.style.display = "none";
        let lastMsg = document.createElement("p");
        lastMsg.textContent = "Thanks for using our Chatbot!";
        lastMsg.classList.add("lastMessage");
        document.body.appendChild(lastMsg);
    }
}

// Function to enable expansion panels
function initializeExpansionPanels() {
    document.querySelectorAll(".sub-item").forEach((item) => {
        item.addEventListener("click", function () {
            const content = this.nextElementSibling;
            const toggleBtn = this.querySelector(".toggle-btn");

            if (content.style.display === "none" || content.style.display === "") {
                content.style.display = "block";
                toggleBtn.innerHTML = "▼";
            } else {
                content.style.display = "none";
                toggleBtn.innerHTML = "▶";
            }
        });
    });
}

// Function to initialize chatbot (Modify if needed)
function initializeChatbot() {
    console.log("Chatbot initialized"); // Debugging message
}

document.addEventListener("DOMContentLoaded", function () {
    initializeExpansionPanels();
    initializeChatbot();
});

/* Open the Pop-Up */
function openModal() {
    document.getElementById("popup-modal").style.display = "flex";
}

/* Close the Pop-Up */
function closeModal() {
    document.getElementById("popup-modal").style.display = "none";
}

/* Save Contact */
function saveContact() {
    // let name = document.getElementById("contact-name").value.trim();
    // let role = document.getElementById("contact-role").value.trim();


    // if (name === "" || role === "") {
    //     alert("Please fill out both fields!");
    //     return;
    // }

    // let contactList = document.querySelector(".contact-list");
    // let newContact = document.createElement("li");
    // newContact.classList.add("contact-item");
    // newContact.innerHTML = `<img src="https://via.placeholder.com/35" alt="${name}"><span>${name} (${role})</span>`;

    // contactList.appendChild(newContact);

    // Close the modal after adding
    closeModal();

    // Clear input fields
    // document.getElementById("contact-name").value = "";
    // document.getElementById("contact-role").value = "";
}


function cancelContact() {

    // Close the modal after adding
    closeModal();

    // Clear input fields
    document.getElementById("contact-name").value = "";
    document.getElementById("contact-role").value = "";
}

document.addEventListener("DOMContentLoaded", function () {
    initializeExpansionPanels();
    initializeChatbot();

    // Ensure buttons exist before adding event listeners
    const importBtn = document.getElementById("importBtn");
    const addBtn = document.getElementById("addBtn");

    if (importBtn) {
        importBtn.addEventListener("click", importData);
    }
    if (addBtn) {
        addBtn.addEventListener("click", openModal);
    }
});

/* Function for IMPORT Button */
function importData() {
    alert("Import function triggered! (Implement import logic here)");
}



document.addEventListener("DOMContentLoaded", function () {
    const fileUpload = document.getElementById("fileUpload");
    const sendChatBtn = document.getElementById("sendBTN");

    fileUpload.addEventListener("change", handleFileUpload);
    sendChatBtn.addEventListener("click", handleChat);

    function handleFileUpload(event) {
        const file = event.target.files[0];
        if (file) {
            // Create a new chat message to show the file name
            const chatbox = document.querySelector(".chatbox");
            const fileMessage = document.createElement("li");
            fileMessage.classList.add("chat", "chat-outgoing");
            fileMessage.innerHTML = `<p>📎 ${file.name} uploaded</p>`;
            chatbox.appendChild(fileMessage);
        }
    }

    function handleChat() {
        let userMessage = document.querySelector(".chat-input textarea").value.trim();
        if (!userMessage) return;

        // Display user message
        const chatbox = document.querySelector(".chatbox");
        const userChat = document.createElement("li");
        userChat.classList.add("chat", "chat-outgoing");
        userChat.innerHTML = `<p>${userMessage}</p>`;
        chatbox.appendChild(userChat);

        // Clear input field
        document.querySelector(".chat-input textarea").value = "";

        // Simulate file sending (Modify this for API integration)
        const file = fileUpload.files[0];
        if (file) {
            console.log("File ready to send:", file);
            // TODO: Implement actual file sending to OpenAI or backend.
        }
    }
});