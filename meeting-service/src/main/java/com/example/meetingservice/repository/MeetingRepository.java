package com.example.meetingservice.repository;

import com.example.meetingservice.entity.Meeting;
import org.springframework.data.jpa.repository.JpaRepository;

public interface MeetingRepository extends JpaRepository<Meeting,Long> {

}