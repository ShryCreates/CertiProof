import { createRoot } from "react-dom/client";
import App from "./App.tsx";
import "./index.css";
// If you want to start measuring performance in your app, pass a function
// to log results (for example: reportWebVitals(console.log))
// or send to an analytics endpoint. Learn more: https://bit.ly/CRA-vitals
createRoot(document.getElementById("root")!).render(<App />);
