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
const generateResponse = (incomingChatLi, userMessage) => {
    const messageElement = incomingChatLi.querySelector("p");

    // System message providing additional context
    const systemMessage = `Answer the next question related to this: 
    There is a person named Gary who is very good at art and design. Athena and
    Chen, who is a INTJ are working on UI. Gary majors in film and design at UPenn.
    He is a freelancer doing UI/UX for many companies.
    `;

    // Combine system instruction with the user's query
    const modifiedUserMessage = `${systemMessage}\n\nUser: ${userMessage}`;

    // API Request
    fetch(API_URL, {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
            "Authorization": `Bearer ${API_KEY}`
        },
        body: JSON.stringify({
            model: "gpt-3.5-turbo",
            messages: [{ role: "user", content: modifiedUserMessage }]
        })
    })
    .then(res => {
        if (!res.ok) throw new Error(`API request failed with status ${res.status}`);
        return res.json();
    })
    .then(data => {
        // Set the response text inside the chat message
        messageElement.textContent = data.choices[0].message.content;
    })
    .catch(error => {
        console.error("Chatbot API Error:", error);
        messageElement.classList.add("error");
        messageElement.textContent = "❌ Error: Please try again!";
    })
    .finally(() => {
        // Scroll the chatbox to the latest message
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

document.addEventListener("DOMContentLoaded", function () {
    const newsItems = document.querySelectorAll(".news-item");
    const toggleNewsBtn = document.getElementById("toggleNewsBtn");

    let currentNewsIndex = 0;

    toggleNewsBtn.addEventListener("click", function () {
        // Hide the current news item
        newsItems[currentNewsIndex].style.display = "none";

        // Move to the next news item (looping back if at the end)
        currentNewsIndex = (currentNewsIndex + 1) % newsItems.length;

        // Show the new current news item
        newsItems[currentNewsIndex].style.display = "block";
    });
});


document.addEventListener("DOMContentLoaded", function () {
    const recWrappers = document.querySelectorAll(".recommendations-wrapper");
    const refreshBtn = document.getElementById("refreshRecommendationsBtn");

    let currentIndex = 0; // Start with the first recommendations list

    refreshBtn.addEventListener("click", function () {
        // Hide all recommendation wrappers
        recWrappers.forEach(wrapper => wrapper.style.display = "none");

        // Toggle to the next wrapper
        currentIndex = (currentIndex + 1) % recWrappers.length;
        recWrappers[currentIndex].style.display = "flex"; // Show the next one
    });

    // Ensure only the first wrapper is visible on page load
    recWrappers.forEach((wrapper, index) => {
        wrapper.style.display = index === 0 ? "flex" : "none";
    });
});


document.getElementById("toggleNewsBtn").addEventListener("click", function() {
    let extraNews = document.getElementById("extraNews");

    if (extraNews.style.display === "none" || extraNews.classList.contains("hidden-news")) {
        extraNews.style.display = "block";
        extraNews.classList.remove("hidden-news");
        this.innerHTML = "🔄 Show Less";
    } else {
        extraNews.style.display = "none";
        extraNews.classList.add("hidden-news");
        this.innerHTML = "🔄 Show More";
    }
});
