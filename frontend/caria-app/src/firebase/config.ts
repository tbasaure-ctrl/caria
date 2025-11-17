/**
 * Firebase Configuration
 * Configuración de Firebase para Wise Adviser
 */

import { initializeApp, FirebaseApp } from 'firebase/app';
import { getAuth, Auth } from 'firebase/auth';
import { getMessaging, Messaging, getToken } from 'firebase/messaging';
import { getAnalytics, Analytics } from 'firebase/analytics';

// Configuración de Firebase
const firebaseConfig = {
  apiKey: "AIzaSyDtVqSNqVRTvHh75RBGYPgsghlKM_9MXcw",
  authDomain: "caria-9b633.firebaseapp.com",
  projectId: "caria-9b633",
  storageBucket: "caria-9b633.firebasestorage.app",
  messagingSenderId: "828514789590",
  appId: "1:828514789590:web:699ddaf60986420dd60177",
  measurementId: "G-CWDKMBK81V"
};

// Inicializar Firebase
let app: FirebaseApp;
let auth: Auth;
let messaging: Messaging | null = null;
let analytics: Analytics | null = null;

try {
  app = initializeApp(firebaseConfig);
  auth = getAuth(app);
  
  // Inicializar Analytics solo en el cliente (no en SSR)
  if (typeof window !== 'undefined') {
    try {
      analytics = getAnalytics(app);
    } catch (error) {
      console.warn('Firebase Analytics no disponible:', error);
    }
  }
  
  // Inicializar Messaging solo si está disponible (requiere HTTPS o localhost)
  if (typeof window !== 'undefined' && 'serviceWorker' in navigator) {
    try {
      messaging = getMessaging(app);
    } catch (error) {
      console.warn('Firebase Messaging no disponible:', error);
    }
  }
} catch (error) {
  console.error('Error inicializando Firebase:', error);
  throw error;
}

// Función para obtener FCM token (para notificaciones push)
export async function getFCMToken(): Promise<string | null> {
  if (!messaging) {
    console.warn('Firebase Messaging no está disponible');
    return null;
  }

  try {
    // Necesitas registrar un service worker primero
    // Ver: FIREBASE_FCM_SETUP.md
    const token = await getToken(messaging, {
      vapidKey: 'TU_VAPID_KEY_AQUI' // Obtén esto de Firebase Console → Cloud Messaging → Web configuration
    });
    return token;
  } catch (error) {
    console.error('Error obteniendo FCM token:', error);
    return null;
  }
}

export { app, auth, messaging, analytics };
export default app;

