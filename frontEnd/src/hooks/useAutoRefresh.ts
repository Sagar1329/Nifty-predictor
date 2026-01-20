import {useEffect,useState} from 'react';

export function useAutoRefresh(intervalMs: number){
    const [secondsLeft, setSecondsLeft] = useState(intervalMs / 1000);
    useEffect(() => {
        setSecondsLeft(intervalMs / 1000);
        const countdown = setInterval(() => {
            setSecondsLeft((prev)=> (prev >1 ? prev-1 : intervalMs / 1000));
        },1000);
        return () => clearInterval(countdown);
    },[intervalMs]);
    return secondsLeft;
}