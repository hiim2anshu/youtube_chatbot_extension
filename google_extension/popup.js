
document.getElementById('summarize-btn').addEventListener('click', async () => {
  const prompt = document.getElementById('prompt-input').value;
  const resultText = document.getElementById('result-text');
  const loading = document.getElementById('loading');

  resultText.textContent = '';
  loading.style.display = 'block';

  try {
    // Get current active tab
    const tabs = await chrome.tabs.query({ active: true, currentWindow: true });
    const url = tabs[0].url;
    const videoId = new URL(url).searchParams.get('v');

    if (!videoId) {
      resultText.textContent = "Please navigate to a YouTube video page first.";
      loading.style.display = 'none';
      return;
    }

    // Backend URL
    const backendUrl = 'http://localhost:5000/process-video';

    const response = await fetch(backendUrl, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ video_id: videoId, prompt: prompt })
    });

    const data = await response.json();

    if (response.ok) {
      resultText.textContent = data.summary;
    } else {
      resultText.textContent = data.error || "An error occurred.";
    }

  } catch (error) {
    console.error('Error:', error);
    resultText.textContent = "An error occurred while connecting to the backend.";
  } finally {
    loading.style.display = 'none';
  }
});
