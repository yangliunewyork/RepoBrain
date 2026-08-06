package com.example.model;

import java.util.Objects;

/**
 * A simple widget with a name and price.
 */
public class Widget {
    private final String name;
    private final double price;

    public Widget(String name, double price) {
        this.name = name;
        this.price = price;
    }

    /**
     * Returns the widget's display name.
     */
    public String getName() {
        return name;
    }

    public double getPrice() {
        return price;
    }

    @Override
    public boolean equals(Object other) {
        if (!(other instanceof Widget)) {
            return false;
        }
        Widget w = (Widget) other;
        return Objects.equals(name, w.name);
    }
}
