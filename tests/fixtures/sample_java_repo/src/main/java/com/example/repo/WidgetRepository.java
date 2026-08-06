package com.example.repo;

import com.example.model.Widget;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

/**
 * In-memory storage for {@link Widget} instances.
 */
public interface WidgetRepository {
    void save(Widget widget);

    Optional<Widget> findByName(String name);

    List<Widget> findAll();
}
